const $ = (id) => document.getElementById(id);
const textElement = (tag, text, cls) => { const el=document.createElement(tag); el.textContent=text; if(cls)el.className=cls; return el; };
const sourceLink = (hit) => { const a=textElement('a',hit.title+' · '+hit.section); const url=new URL(hit.url); if(url.protocol==='https:'&&url.hostname==='www.nidcr.nih.gov'){a.href=url.href;a.target='_blank';a.rel='noopener noreferrer';} return a; };
fetch('/api/health').then(r=>{if(!r.ok)throw new Error();return r.json();}).then(d=>{
  $('status').textContent=`${d.chunks} source passages · ${d.backend === 'semantic' ? 'Semantic models' : 'SMOKE MODE — lexical test approximation'} · ${d.generator} answers`;
}).catch(()=>{$('status').textContent='Pipeline unavailable. Check the local server.';});
document.querySelectorAll('.example').forEach(button=>button.addEventListener('click',()=>{$('question').value=button.textContent;$('question').focus();}));
$('query-form').addEventListener('submit',async(event)=>{
  event.preventDefault();$('error').hidden=true;$('result').hidden=true;$('submit').disabled=true;$('submit').textContent='Retrieving…';
  try{
    const response=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:$('question').value,strategy:$('strategy').value,top_k:3})});
    const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Please enter a valid question.');
    $('answer').textContent=data.answer;$('answer-mode').textContent=data.abstained?'ABSTAINED':data.generator.toUpperCase();$('raw').textContent=JSON.stringify(data,null,2);
    $('citations').replaceChildren();data.hits.filter(h=>data.cited_ids.includes(h.id)).forEach(h=>$('citations').append(sourceLink(h)));
    $('stats').replaceChildren(...[`${data.retrieval_ms.toFixed(1)} ms retrieval`,`${data.total_ms.toFixed(1)} ms total`,`${data.hits.length} passages`,`Trace ${data.trace_id.slice(0,10)}`].map(t=>textElement('span',t)));
    $('hits').replaceChildren();data.hits.forEach((hit,i)=>{const card=document.createElement('article');card.className='hit'+(data.cited_ids.includes(hit.id)?' cited':'');const heading=textElement('div','', 'hit-heading');heading.append(sourceLink(hit),textElement('span',`#${i+1} · score ${hit.score.toFixed(4)}`));card.append(heading,textElement('p',hit.text),textElement('small',hit.id+' · '+(data.cited_ids.includes(hit.id)?'cited in answer':'retrieved context')));$('hits').append(card);});
    $('result').hidden=false;$('result').scrollIntoView({behavior:'smooth',block:'start'});
  }catch(error){$('error').textContent=error.message;$('error').hidden=false;}
  finally{$('submit').disabled=false;$('submit').textContent='Find evidence ↗';}
});
