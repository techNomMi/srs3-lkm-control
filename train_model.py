"""СРС 3. Учебные данные, регрессия и сравнение управления.

Запуск: python train_model.py. Требуется только NumPy.
Все коэффициенты генератора и установки условные, не производственные.
"""
from pathlib import Path
import csv
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parent
SEED = 20261102
FEATURES = ['T_C', 'n_rpm', 't_min', 'C_percent']
PARAMS = dict(m=50.0, cp=2000.0, UA=15.0, ambient=22.0,
              heater_max=1500.0, mixing_base_kw=0.4, heat_fraction=0.2,
              dt=0.25, max_time=20.0, k=0.001, Tset=35.0,
              mu_min=450.0, mu_max=550.0, Hmin=95.0,
              Tmin=20.0, Tmax=45.0, forecast=3.0)


def true_mu(T, n, t, C, batch_offset=0):
    """Синтетический объект. Измеренный отклик дополнительно содержит шум."""
    return 620 - 8*(T-25) - .12*(n-300) - 4*t + 8*(C-40) + .015*(T-25)*(C-40) + batch_offset


def fit(X, y):
    # Оценка МНК через устойчивое решение, без обращения X^T X.
    center, scale = X.mean(axis=0), X.std(axis=0)
    if np.any(scale == 0):
        raise ValueError('Для обучения необходим разброс всех признаков.')
    Z = np.column_stack([np.ones(len(X)), (X-center)/scale])
    beta, _, rank, _ = np.linalg.lstsq(Z, y, rcond=None)
    if rank != Z.shape[1]:
        raise ValueError('Недостаточный ранг матрицы признаков.')
    slopes = beta[1:]/scale
    return np.r_[beta[0] - center@slopes, slopes]


def predict(coef, T, n, t, C):
    return float(np.array([1, T, n, t, C])@coef)


def metrics(y, p):
    error = y-p
    return dict(MAE=float(np.abs(error).mean()), RMSE=float(np.sqrt((error**2).mean())),
                R2=float(1-(error**2).sum()/((y-y.mean())**2).sum()))


def power(n):
    return PARAMS['mixing_base_kw']*(n/300)**3


def next_T(T, n, heater, minutes):
    a = PARAMS
    # Время в минутах переводится в секунды; мощности в Вт.
    heat = heater + a['heat_fraction']*power(n)*1000
    return T + 60*minutes*(heat-a['UA']*(T-a['ambient']))/(a['m']*a['cp'])


def forecast_T(T, n, heater, minutes):
    # Прогноз с тем же шагом, что у имитационного объекта.
    for _ in range(round(minutes/PARAMS['dt'])):
        T = next_T(T,n,heater,PARAMS['dt'])
    return T


def valid_sensor(T, C):
    return math.isfinite(T) and math.isfinite(C) and 20 <= T <= 45 and 30 <= C <= 50


def choose(T, t, C, dose, coef):
    """Перебор конечного множества воздействий, горизонт 3 мин.

    Это упрощённый прогнозный регулятор, не промышленная MPC-система.
    """
    if not valid_sensor(T,C):
        return 0,0  # Безопасная остановка в рамках учебной модели.
    horizon = min(PARAMS['forecast'], PARAMS['max_time']-t)
    candidates=[]
    for n in [100,200,300,400,500]:
        for heater in [0,500,1000,1500]:
            Tf=forecast_T(T,n,heater,horizon)
            if not 20 <= Tf <= 45:
                continue
            mu=predict(coef,Tf,n,t+horizon,C)
            H=-100*math.expm1(-PARAMS['k']*(dose+n*horizon))
            violation=max(450-mu,0,mu-550)/50
            J=5*violation**2 + 8*(max(95-H,0)/5)**2 + .02*((mu-500)/50)**2 + .35*(power(n)+heater/1000)*horizon/60
            candidates.append((J,n,heater))
    if not candidates:
        return 0,0
    _,n,heater=min(candidates)
    return n,heater


def simulate(coef, mode, initial_T=25.0, C=40.0, batch_offset=0.0):
    if mode not in ['traditional','intelligent']:
        raise ValueError('Неизвестный режим управления.')
    if not valid_sensor(initial_T,C):
        return dict(mode=mode,success=False,status='sensor_error',time=0,energy=0,trace=[])
    T,energy,dose=initial_T,0.0,0.0
    trace=[]
    a=PARAMS
    reached=False
    for step in range(round(a['max_time']/a['dt'])+1):
        t=step*a['dt']
        # Выбор управления использует датчики и обученный прогноз, не true_mu.
        if mode=='traditional':
            n=300
            heater=max(0,min(1500,450*(a['Tset']-T)))
        else:
            n,heater=choose(T,t,C,dose,coef)
        mu=true_mu(T,n,t,C,batch_offset)
        pred=predict(coef,T,n,t,C)
        H=-100*math.expm1(-a['k']*dose)
        trace.append(dict(t=t,T=T,n=n,heater=heater,mu=mu,pred=pred,H=H,energy=energy))
        # Качество проверяется по независимому модельному датчику вязкости.
        reached=450 <= mu <= 550 and H >= 95 and 20 <= T <= 45
        if reached or t>=a['max_time']:
            break
        if not valid_sensor(T,C) or n==0:
            break
        Tnext=next_T(T,n,heater,a['dt'])
        if Tnext>45 or Tnext<20:
            break
        energy+=(power(n)+heater/1000)*a['dt']/60
        dose+=n*a['dt']
        T=Tnext
    return dict(mode=mode,success=bool(reached),status='ready' if reached else 'not_ready',
                time=trace[-1]['t'],energy=energy,
                temp_MAE=float(np.mean([abs(x['T']-35) for x in trace])),
                mu_final=trace[-1]['mu'],H_final=trace[-1]['H'],
                prediction_MAE=float(np.mean([abs(x['pred']-x['mu']) for x in trace])),trace=trace)


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    rng=np.random.default_rng(SEED)
    rows=[]
    # 100 партий, по 8 наблюдений. Смещение партии создаёт зависимость строк.
    for batch in range(100):
        offset=rng.normal(0,5)
        for j in range(8):
            T,n,t,C=rng.uniform(20,45),rng.uniform(100,500),rng.uniform(0,20),rng.uniform(30,50)
            mu=true_mu(T,n,t,C,offset)+rng.normal(0,8)
            rows.append(dict(batch=batch,T_C=T,n_rpm=n,t_min=t,C_percent=C,mu_mPas=mu))
    # Деление по целым партиям: одна партия не попадает в обе выборки.
    groups=rng.permutation(100);train_groups=set(groups[:80].tolist())
    for r in rows:r['split']='train' if r['batch'] in train_groups else 'test'
    train=[r for r in rows if r['split']=='train'];test=[r for r in rows if r['split']=='test']
    X=np.array([[r[k] for k in FEATURES] for r in train]);y=np.array([r['mu_mPas'] for r in train])
    Xt=np.array([[r[k] for k in FEATURES] for r in test]);yt=np.array([r['mu_mPas'] for r in test])
    coef=fit(X,y);pred=np.column_stack([np.ones(len(Xt)),Xt])@coef
    test_result=[dict(**r,predicted=float(p),residual=float(r['mu_mPas']-p)) for r,p in zip(test,pred)]
    baseline=np.full(len(yt),y.mean())
    summary={k:dict(min=float(min(r[k] for r in rows)),max=float(max(r[k] for r in rows)),mean=float(np.mean([r[k] for r in rows])),std=float(np.std([r[k] for r in rows]))) for k in FEATURES+['mu_mPas']}
    corr=np.corrcoef(np.array([[r[k] for k in FEATURES+['mu_mPas']] for r in rows]).T)[-1,:-1].tolist()
    # Набор проверки регуляторов отделён от обучения и теста регрессии.
    ctrl_rng=np.random.default_rng(SEED+1)
    comparisons=[]
    for i in range(30):
        T0=float(ctrl_rng.uniform(22,30));C=float(ctrl_rng.uniform(36,44));off=float(ctrl_rng.normal(0,5))
        pair=[simulate(coef,mode,T0,C,off) for mode in ['traditional','intelligent']]
        for r in pair:comparisons.append(dict(case=i+1,T0=T0,C=C,**{k:v for k,v in r.items() if k!='trace'}))
    aggregates={}
    for mode in ['traditional','intelligent']:
        selected=[r for r in comparisons if r['mode']==mode]
        good=[r for r in selected if r['success']]
        aggregates[mode]=dict(cases=len(selected),passed=len(good),quality_percent=100*len(good)/len(selected),
                              energy_all=float(np.mean([r['energy'] for r in selected])),
                              time_success=float(np.mean([r['time'] for r in good])) if good else None,
                              temp_MAE=float(np.mean([r['temp_MAE'] for r in selected])),
                              prediction_MAE=float(np.mean([r['prediction_MAE'] for r in selected])))
    matched=[]
    for i in range(30):
        a,b=comparisons[2*i:2*i+2]
        if a['success'] and b['success']:matched.append((a['energy'],b['energy']))
    saving=100*(1-np.mean([x[1] for x in matched])/np.mean([x[0] for x in matched])) if matched else None
    report=dict(seed=SEED,rows=len(rows),train_rows=len(train),test_rows=len(test),train_batches=80,test_batches=20,
                coef=coef.tolist(),features=FEATURES,metrics=metrics(yt,pred),baseline_metrics=metrics(yt,baseline),
                summary=summary,correlations=corr,missing=0,duplicates=0,params=PARAMS,
                test=test_result,preview=rows[:12],aggregates=aggregates,matched_cases=len(matched),matched_energy_saving=saving,
                comparisons=comparisons,demo={mode:simulate(coef,mode) for mode in ['traditional','intelligent']})
    (ROOT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'data.js').write_text('window.REPORT = '+json.dumps(report,ensure_ascii=False)+';\n',encoding='utf-8')
    write_csv(ROOT/'dataset.csv',rows);write_csv(ROOT/'test_predictions.csv',test_result);write_csv(ROOT/'control_comparison.csv',comparisons)
    print(json.dumps({k:report[k] for k in ['rows','metrics','baseline_metrics','aggregates','matched_energy_saving']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
