'use strict';
const R=window.REPORT, $=id=>document.getElementById(id);
const f=(v,d=3)=>v===null?'—':Number(v).toLocaleString('ru-RU',{maximumFractionDigits:d});
const modes={traditional:'Традиционное',intelligent:'Прогнозное'};
function table(id,headers,rows){$(id).innerHTML='<div class="table-scroll"><table><thead><tr>'+headers.map(x=>'<th scope="col">'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
function chart(id,title,series,xlabel,ylabel,options={}){
 const palette=['#286da2','#c06636','#5a8552'];
 const all=series.flatMap(s=>s.points),xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);
 const xmin=options.xmin??Math.min(...xs),xmax=options.xmax??Math.max(...xs);
 const rawMin=options.ymin??Math.min(...ys),rawMax=options.ymax??Math.max(...ys),pad=(rawMax-rawMin||1)*.08;
 const ymin=options.ymin??rawMin-pad,ymax=options.ymax??rawMax+pad;
 const w=800,h=340,l=66,r=20,t=27,b=54,pw=w-l-r,ph=h-t-b;
 const X=x=>l+(x-xmin)/(xmax-xmin||1)*pw,Y=y=>t+ph-(y-ymin)/(ymax-ymin)*ph;
 let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${title}">`;
 for(let i=0;i<=5;i++){const y=ymin+(ymax-ymin)*i/5;svg+=`<line class="grid" x1="${l}" x2="${w-r}" y1="${Y(y)}" y2="${Y(y)}"/><text x="${l-8}" y="${Y(y)+4}" text-anchor="end">${f(y,1)}</text>`;}
 for(let i=0;i<=5;i++){const x=xmin+(xmax-xmin)*i/5;svg+=`<text x="${X(x)}" y="${h-b+20}" text-anchor="middle">${f(x,1)}</text>`;}
 svg+=`<text x="${l}" y="14">${ylabel}</text><text x="${l+pw/2}" y="${h-6}" text-anchor="middle">${xlabel}</text>`;
 for(const bound of options.bounds||[]){svg+=`<line x1="${l}" x2="${w-r}" y1="${Y(bound)}" y2="${Y(bound)}" stroke="#888" stroke-dasharray="4 4"/>`;}
 series.forEach((s,i)=>{svg+=`<path class="line" stroke="${palette[i%3]}" d="${s.points.map(([x,y],j)=>`${j?'L':'M'}${X(x).toFixed(2)},${Y(y).toFixed(2)}`).join(' ')}"/>`;if(options.dots)s.points.forEach(([x,y])=>svg+=`<circle cx="${X(x)}" cy="${Y(y)}" r="2" fill="${palette[i%3]}"/>`);});
 svg+=`<line class="cursor" y1="${t}" y2="${h-b}" stroke="#8b999e" visibility="hidden"/><rect class="hit" x="${l}" y="${t}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;
 const el=$(id);el.className='chart';el.innerHTML=`<h3>${title}</h3>${svg}<div class="legend">${series.map((s,i)=>`<span><i style="background:${palette[i%3]}"></i>${s.name}</span>`).join('')}</div><div class="tooltip">Для значений наведите указатель или коснитесь графика.</div>`;
 const image=el.querySelector('svg'),hit=el.querySelector('.hit'),cursor=el.querySelector('.cursor');
 function move(e){const box=image.getBoundingClientRect(),x=Math.max(xmin,Math.min(xmax,xmin+((e.clientX-box.left)*w/box.width-l)/pw*(xmax-xmin)));cursor.setAttribute('x1',X(x));cursor.setAttribute('x2',X(x));cursor.setAttribute('visibility','visible');el.querySelector('.tooltip').textContent=`${xlabel}: ${f(x,2)} · `+series.map(s=>{if(x>s.points.at(-1)[0])return s.name+': опыт завершён';const p=s.points.reduce((a,b)=>Math.abs(b[0]-x)<Math.abs(a[0]-x)?b:a);return s.name+': '+f(p[1],3);}).join(' · ');}
 hit.addEventListener('pointermove',move);hit.addEventListener('pointerdown',move);
}
function renderReport(){
 const a=R.params;
 table('params-table',['Параметр','Значение','Пояснение'],[['m',a.m+' кг','Масса смеси'],['Cₚ',a.cp+' Дж/(кг·°C)','Теплоёмкость'],['UA',a.UA+' Вт/°C','Коэффициент теплообмена'],['Tср',a.ambient+' °C','Температура среды'],['η',a.heat_fraction,'Доля мощности мешалки, переходящая в тепло'],['Pₕ,max',a.heater_max+' Вт','Мощность нагревателя'],['Δt',a.dt+' мин','Шаг интегрирования Эйлером'],['tmax',a.max_time+' мин','Предел длительности']]);
 $('split-info').textContent=`Всего ${R.rows} строк: ${R.train_rows} для обучения (${R.train_batches} партий), ${R.test_rows} для теста (${R.test_batches} партий). Seed = ${R.seed}. Все данные синтетические.`;
 $('eda').textContent=`Пропусков: ${R.missing}; полных дубликатов: ${R.duplicates}. Выбросы автоматически не удалялись. Во всех наблюдениях концентрации и технологические параметры находятся в заданных диапазонах.`;
 const labels={'T_C':'T, °C','n_rpm':'n, об/мин','t_min':'t, мин','C_percent':'C, %','mu_mPas':'μ, мПа·с'};
 table('stats',['Признак','Минимум','Максимум','Среднее','Ст. отклонение'],Object.entries(R.summary).map(([k,v])=>[labels[k],f(v.min),f(v.max),f(v.mean),f(v.std)]));
 $('correlations').textContent='Корреляция с вязкостью: '+R.features.map((k,i)=>labels[k]+': '+f(R.correlations[i],3)).join('; ')+'.';
 const c=R.coef;$('regression-formula').textContent=`μ̂ = ${f(c[0],6)} ${c.slice(1).map((v,i)=>(v>=0?'+ ':'− ')+f(Math.abs(v),6)+' · '+['T','n','t','C'][i]).join(' ')}`;
 table('metrics',['Модель','MAE, мПа·с','RMSE, мПа·с','R²'],[['Линейная регрессия',f(R.metrics.MAE),f(R.metrics.RMSE),f(R.metrics.R2,5)],['Среднее обучающей выборки',f(R.baseline_metrics.MAE),f(R.baseline_metrics.RMSE),f(R.baseline_metrics.R2,5)]]);
 chart('prediction-chart','Факт и прогноз: все тестовые наблюдения',[{name:'Модельное измерение',points:R.test.map((r,i)=>[i+1,r.mu_mPas])},{name:'Прогноз',points:R.test.map((r,i)=>[i+1,r.predicted])}],'Номер наблюдения','Вязкость, мПа·с',{dots:true});
 chart('residual-chart','Ошибка: измерение минус прогноз',[{name:'Остаток',points:R.test.map((r,i)=>[i+1,r.residual])}],'Номер наблюдения','Ошибка, мПа·с',{bounds:[0],dots:true});
 table('test-table',['Партия','T, °C','n, об/мин','t, мин','C, %','Факт μ','Прогноз μ','Ошибка'],R.test.slice(0,20).map(r=>[r.batch,f(r.T_C),f(r.n_rpm),f(r.t_min),f(r.C_percent),f(r.mu_mPas),f(r.predicted),f(r.residual)]));
 table('demo-table',['Регулятор','Готовность','Время, мин','Энергия, кВт·ч','Конечная μ','Конечная H, %'],Object.values(R.demo).map(r=>[modes[r.mode],r.success?'Готово':'Не готово',f(r.time,2),f(r.energy,4),f(r.mu_final),f(r.H_final)]));
 for(const [id,title,key,ylabel,opts] of [['mu-control','Вязкость при двух способах управления','mu','μ, мПа·с',{bounds:[450,550]}],['T-control','Температура смеси','T','T, °C',{ymin:20,ymax:45,bounds:[35]}],['n-control','Скорость мешалки','n','n, об/мин',{ymin:0,ymax:550}],['H-control','Однородность смеси','H','H, %',{ymin:0,ymax:100,bounds:[95]}],['E-control','Накопленное энергопотребление','energy','E, кВт·ч',{}]]){
  chart(id,title,Object.values(R.demo).map(r=>({name:modes[r.mode],points:r.trace.map(s=>[s.t,s[key]])})),'Время, мин',ylabel,opts);
 }
 table('aggregate-table',['Критерий','Традиционное','Прогнозное'],[
 ['Готовые партии, шт.',...Object.values(R.aggregates).map(r=>r.passed+' / '+r.cases)],
 ['Доля годных, %',...Object.values(R.aggregates).map(r=>f(r.quality_percent,2))],
 ['Средняя энергия всех партий, кВт·ч',...Object.values(R.aggregates).map(r=>f(r.energy_all,4))],
 ['Среднее время успешных партий, мин',...Object.values(R.aggregates).map(r=>f(r.time_success,3))],
 ['Среднее |T − 35|, °C',...Object.values(R.aggregates).map(r=>f(r.temp_MAE,3))],
 ['Средняя ошибка прогноза на траектории, мПа·с',...Object.values(R.aggregates).map(r=>f(r.prediction_MAE,3))]]);
 $('matched-result').className='result';$('matched-result').textContent=`В ${R.matched_cases} парных сценариях оба регулятора достигли качества. Изменение средней энергии прогнозного регулятора относительно традиционного: ${f(-R.matched_energy_saving,2)}%. Положительная экономия в этом наборе: ${f(R.matched_energy_saving,2)}%.`;
 table('comparison-table',['Сценарий','T₀','C, %','Регулятор','Готовность','t, мин','E, кВт·ч','μ','H, %'],R.comparisons.map(r=>[r.case,f(r.T0,2),f(r.C,2),modes[r.mode],r.success?'Да':'Нет',f(r.time,2),f(r.energy,4),f(r.mu_final,2),f(r.H_final,2)]));
 const tr=R.aggregates.traditional,it=R.aggregates.intelligent;
 $('conclusions').innerHTML=`<p>Линейная регрессия на независимом тесте дала MAE ${f(R.metrics.MAE,2)} мПа·с и R² ${f(R.metrics.R2,4)}. Она точнее базового прогноза средним значением на данном синтетическом наборе.</p><p>В контрольных сценариях традиционный регулятор завершил ${tr.passed} партий из ${tr.cases}, прогнозный — ${it.passed} из ${it.cases}. Средняя энергия составила ${f(tr.energy_all,4)} и ${f(it.energy_all,4)} кВт·ч соответственно. Парная оценка экономии — ${f(R.matched_energy_saving,2)}%.</p><p>Среднее время успешной обработки: ${f(tr.time_success,3)} и ${f(it.time_success,3)} мин. Отклонение от 35 °C: ${f(tr.temp_MAE,3)} и ${f(it.temp_MAE,3)} °C. Прогнозный регулятор допускает температурное отклонение внутри разрешённого диапазона ради других критериев.</p><p>Результаты зависят от выбранного генератора, коэффициентов объекта, весов регулятора и набора сценариев. Не учтены масштабирование установки, неопределённость теплообмена, скорость реакции исполнительных механизмов и отказы датчиков. Экономию и качество нельзя переносить на производство без испытаний.</p>`;
}
function forecast(){
 const ids=['fT','fn','ft','fC'];const invalid=ids.find(id=>!$(id).checkValidity());
 if(invalid){$('forecast-output').className='result error';$('forecast-output').textContent='Введите все четыре параметра в указанных диапазонах. Экстраполяция за область обучения заблокирована.';return;}
 const x=ids.map(id=>Number($(id).value));const mu=R.coef[0]+x.reduce((sum,v,i)=>sum+v*R.coef[i+1],0);
 const H=-100*Math.expm1(-R.params.k*x[1]*x[2]);const ok=mu>=450&&mu<=550;
 $('forecast-output').className='result';$('forecast-output').innerHTML=`<b>Прогноз вязкости: ${f(mu,2)} мПа·с.</b><p>${ok?'Прогноз внутри диапазона 450–550 мПа·с.':'Прогноз вне диапазона 450–550 мПа·с.'} Расчётная H при постоянной скорости: ${f(H,2)}%.</p><p>Тестовая RMSE модели: ${f(R.metrics.RMSE,2)} мПа·с. Это средняя характеристика ошибки на синтетическом тесте, не доверительный интервал и не гарантия качества отдельной партии.</p>`;
}
function route(){const id=location.hash.slice(1)||'title',page=document.getElementById(id);if(!page||!page.classList.contains('page')){location.hash='title';return;}document.querySelectorAll('.page').forEach(s=>s.hidden=s.id!==id);document.querySelectorAll('nav a').forEach(a=>{const active=a.hash==='#'+id;a.classList.toggle('active',active);active?a.setAttribute('aria-current','page'):a.removeAttribute('aria-current');});window.scrollTo(0,0);}
$('forecast-form').addEventListener('submit',e=>{e.preventDefault();forecast();});$('forecast-form').addEventListener('input',forecast);window.addEventListener('hashchange',route);
renderReport();forecast();route();
