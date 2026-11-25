const app = document.getElementById("app");
const PINK = "#f50087";
let charts = [];
let sceneCleanup = null;

const fmt = (n, d=0) => Number(n).toLocaleString("en-US", {
  maximumFractionDigits:d,
  minimumFractionDigits:d
});
const money = (n) => "R$ " + fmt(n, 0);
const pct = (n, d=2) => (100*n).toFixed(d) + "%";

async function api(path) {
  const response = await fetch("/api/" + path);
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "API unavailable");
  return result;
}

function kpi(label, value, hint="", pink=false) {
  return `<div class="card kpi ${pink?"pink-card":""}">
    <span class="label">${label}</span>
    <div class="value">${value}</div>
    <span class="hint">${hint}</span>
  </div>`;
}

function heading(tag, title, subtitle) {
  return `<div class="eyebrow">${tag}</div>
    <h1>${title}</h1><p class="sub">${subtitle}</p>`;
}

function makeChart(id, type, labels, data, color=PINK) {
  const el = document.getElementById(id);
  if (!el || typeof Chart === "undefined") return;
  charts.push(new Chart(el, {
    type,
    data: {
      labels,
      datasets: [{
        data,
        borderColor:color,
        backgroundColor:type==="line"?"#f5008720":color,
        borderWidth:3,
        fill:type==="line",
        tension:.36,
        pointRadius:type==="line"?2:0,
        borderRadius:5
      }]
    },
    options: {
      responsive:true,
      maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{grid:{display:false},ticks:{maxTicksLimit:8}},
        y:{beginAtZero:true,grid:{color:"#f0f0f2"}}
      }
    }
  }));
}

function barRows(items, nameKey, valueKey, formatter=fmt) {
  const max = Math.max(...items.map(x=>x[valueKey]),1);
  return `<div class="bars">${items.map(x=>`
    <div class="barline">
      <span>${x[nameKey]}</span>
      <div class="track"><div class="fill" style="width:${100*x[valueKey]/max}%"></div></div>
      <b>${formatter(x[valueKey])}</b>
    </div>`).join("")}</div>`;
}

function clearPage() {
  charts.forEach(c=>c.destroy());
  charts=[];
  if(sceneCleanup) {sceneCleanup();sceneCleanup=null;}
}

function scene3D(id, values, kind="columns") {
  if(typeof THREE==="undefined") return;
  const holder=document.getElementById(id);
  if(!holder) return;

  const width=holder.clientWidth, height=holder.clientHeight;
  const scene=new THREE.Scene();
  const camera=new THREE.PerspectiveCamera(40,width/height,.1,100);
  camera.position.set(5,5,9);
  camera.lookAt(0,0,0);
  const renderer=new THREE.WebGLRenderer({alpha:true,antialias:true});
  renderer.setSize(width,height);
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  holder.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff,2));
  const light=new THREE.DirectionalLight(0xffffff,3);
  light.position.set(4,7,5);
  scene.add(light);

  const max=Math.max(...values,1);
  const group=new THREE.Group();
  scene.add(group);

  values.forEach((v,i)=>{
    const h=.4+2.6*v/max;
    const geometry=kind==="islands"
      ? new THREE.CylinderGeometry(.48,.62,h,40)
      : new THREE.BoxGeometry(.65,h,.65);
    const material=new THREE.MeshStandardMaterial({
      color:i%3===0?0xf50087:i%3===1?0xffffff:0x222222,
      metalness:.12,
      roughness:.32
    });
    const mesh=new THREE.Mesh(geometry,material);
    mesh.position.set((i-(values.length-1)/2)*1.18,h/2-1.2,0);
    group.add(mesh);
  });

  let frame, alive=true;
  function animate(){
    if(!alive)return;
    frame=requestAnimationFrame(animate);
    group.rotation.y+=.002;
    renderer.render(scene,camera);
  }
  animate();

  sceneCleanup=()=>{
    alive=false;
    cancelAnimationFrame(frame);
    renderer.dispose();
    renderer.domElement.remove();
  };
}

async function overview() {
  const d=await api("overview");
  app.innerHTML=heading(
    "MARKETPLACE PERFORMANCE",
    "Understand Your <em>Business.</em>",
    "Explore historical sales, orders and delivery performance."
  )+`
  <div class="grid kpis">
    ${kpi("Total Revenue",money(d.total_revenue),"Historical payment value",true)}
    ${kpi("Total Orders",fmt(d.total_orders),"Unique orders")}
    ${kpi("Average Order Value",money(d.average_order_value),"Revenue per order")}
    ${kpi("Late Delivery Rate",pct(d.historical_late_rate),"Delivered orders with known dates")}
  </div>
  <div class="card" style="margin-bottom:18px">
    <h2>3D Sales Category Landscape</h2>
    <p class="sub">Column height shows category sales. Taller means more revenue.</p>
    <div class="scene" id="overview3d"></div>
    <div class="legend"><span>Higher sales</span><span>Other categories</span><span>Reference columns</span></div>
  </div>
  <div class="grid two">
    <div class="card"><h2>Monthly Sales Trend</h2><div class="chartbox"><canvas id="monthly"></canvas></div></div>
    <div class="card"><h2>Top Categories</h2><div id="categories"></div></div>
  </div>
  <div class="card hero hero-row">
    <div><h2>Which Customers Drive Growth?</h2><p class="sub">Explore spending, repeat purchases and customer groups.</p></div>
    <button class="action" onclick="navigate('customers')">Explore Customers →</button>
  </div>`;
  document.getElementById("categories").innerHTML=
    barRows(d.categories,"name","revenue",n=>"R$"+(n/1e6).toFixed(2)+"M");
  makeChart("monthly","line",d.monthly.map(x=>x.month),d.monthly.map(x=>x.revenue));
  scene3D("overview3d",d.categories.slice(0,6).map(x=>x.revenue));
}

async function customers() {
  const d=await api("customers");
  app.innerHTML=heading(
    "CUSTOMER INTELLIGENCE",
    "Understand Your <em>Customers.</em>",
    "See who spends more, who returns and how customers rate their experience."
  )+`
  <div class="grid kpis">
    ${kpi("Unique Customers",fmt(d.unique_customers),"Customer identities",true)}
    ${kpi("Repeat Purchase Rate",pct(d.repeat_rate),"Customers with multiple orders")}
    ${kpi("Average Review",fmt(d.average_review,2)+" / 5","Historical reviews")}
    ${kpi("Customer Groups",d.segments.length,"Rule-based RFM segments")}
  </div>
  <div class="card" style="margin-bottom:18px">
    <h2>3D Customer Segment Islands</h2>
    <p class="sub">Island height represents customer count. Select a segment below for details.</p>
    <div class="scene" id="customers3d"></div>
    <div class="grid three" id="segments"></div>
  </div>
  <div class="grid two">
    <div class="card"><h2>Customer Group Sizes</h2><div id="segmentbars"></div></div>
    <div class="card"><h2>Review Rating Distribution</h2><div class="chartbox"><canvas id="reviews"></canvas></div></div>
  </div>
  <div class="notice">Segments use transparent RFM rules, not a trained clustering model. Repeat purchase rate is not cohort retention.</div>`;
  document.getElementById("segments").innerHTML=d.segments.map(s=>`
    <div class="result">
      <small>${s.segment}</small>
      <strong>${fmt(s.customers)}</strong>
      <small>Average spend: ${money(s.average_spend)}</small>
    </div>`).join("");
  document.getElementById("segmentbars").innerHTML=
    barRows(d.segments,"segment","customers");
  makeChart("reviews","bar",d.review_scores.map(x=>x.score+" ★"),d.review_scores.map(x=>x.count));
  scene3D("customers3d",d.segments.map(x=>x.customers),"islands");
}

async function predictions() {
  const d=await api("predictions");
  const c=d.confusion;
  app.innerHTML=heading(
    "CHAMPION MODEL AUDIT",
    "Predict Late <em>Deliveries.</em>",
    "Rank risky orders and measure how well the model identifies delays."
  )+`
  <div class="grid two">
    <div class="card dark">
      <h2>3D Risk Ranking</h2>
      <p class="sub">Columns represent flagged late orders, flagged on-time orders and missed late orders.</p>
      <div class="scene" id="risk3d"></div>
      <div class="legend"><span>Late captured</span><span>On-time flagged</span><span>Late missed</span></div>
    </div>
    <div class="card">
      <div class="eyebrow">MAIN VERIFIED RESULT</div>
      <h1><em>${pct(d.recall_at_20,0)}</em></h1>
      <h2>Of late deliveries identified</h2>
      <p class="sub">By reviewing the highest-risk 20% of test orders.</p>
      <div class="grid two" style="margin-top:30px">
        ${kpi("Late Captured",fmt(c.tp), "True positives",true)}
        ${kpi("Lift",fmt(d.lift_at_20,2)+"×","Versus random selection")}
      </div>
    </div>
  </div>
  <div class="grid kpis">
    ${kpi("Evaluated Orders",fmt(d.evaluated_orders),"Historical test set")}
    ${kpi("Actual Late Orders",fmt(d.late_orders),"Ground truth")}
    ${kpi("Precision @20%",pct(d.precision_at_20),"Flagged orders actually late",true)}
    ${kpi("ROC-AUC",fmt(d.roc_auc,4),"Ranking discrimination")}
  </div>
  <div class="grid two">
    <div class="card">
      <h2>Intervention Capacity</h2>
      <p class="sub">How many late orders are found at each review budget?</p>
      <div class="chartbox"><canvas id="capacityChart"></canvas></div>
    </div>
    <div class="card">
      <h2>Confusion Matrix · Top 20%</h2>
      <div class="results">
        <div class="result"><small>Correctly Flagged Late</small><strong class="pink">${fmt(c.tp)}</strong></div>
        <div class="result"><small>On-Time Flagged</small><strong>${fmt(c.fp)}</strong></div>
        <div class="result"><small>Late Missed</small><strong>${fmt(c.fn)}</strong></div>
        <div class="result"><small>Correctly Not Flagged</small><strong>${fmt(c.tn)}</strong></div>
      </div>
    </div>
  </div>
  <div class="notice">Model results are retrospective. Calibration improves probability quality but does not prove future operational performance.</div>`;
  makeChart("capacityChart","line",
    d.capacity.map(x=>(x.capacity*100)+"%"),
    d.capacity.map(x=>x.recall*100)
  );
  scene3D("risk3d",[c.tp,c.fp,c.fn]);
}

async function decisions() {
  app.innerHTML=heading(
    "BUSINESS DECISION ENGINE",
    "Make Better <em>Business Decisions.</em>",
    "Test intervention costs and possible savings before spending."
  )+`
  <div class="grid two">
    <div class="card">
      <h2>Strategic Trade-Off Simulator</h2>
      <p class="sub">Change assumptions to see how costs and benefits change.</p>
      <div class="controls">
        <div class="control"><label>Orders to Check <b id="capacityLabel">20%</b></label>
          <input id="capacity" type="range" min="5" max="50" step="5" value="20"></div>
        <div class="control"><label>Cost per Intervention <b id="costLabel">R$ 8</b></label>
          <input id="cost" type="range" min="1" max="25" step="1" value="8"></div>
        <div class="control"><label>Assumed Success Rate <b id="successLabel">35%</b></label>
          <input id="success" type="range" min="0" max="80" step="5" value="35"></div>
        <div class="control"><label>Value of Avoiding a Delay <b id="valueLabel">R$ 45</b></label>
          <input id="value" type="range" min="15" max="150" step="5" value="45"></div>
      </div>
      <div class="notice">These are assumptions, not actual measured business savings.</div>
    </div>
    <div class="card">
      <h2>Estimated Business Impact</h2>
      <div class="results" id="decisionResults"></div>
      <div class="scene" id="balance3d" style="margin-top:20px;height:240px"></div>
    </div>
  </div>
  <div class="grid two">
    <div class="card">
      <div class="eyebrow">EXPERIMENT DESIGN</div>
      <h2>Would This Work in Practice?</h2>
      <p class="sub">Run a randomized A/B test before real rollout.</p>
      <div class="results">
        <div class="result"><small>Group A</small><strong>Control</strong><small>Standard delivery process</small></div>
        <div class="result"><small>Group B</small><strong>Treatment</strong><small>Model-guided intervention</small></div>
      </div>
      <div class="notice">Track actual late delivery rate, costs and customer experience. Predefine sample size and stopping rules.</div>
    </div>
    <div class="card">
      <div class="eyebrow">BUSINESS ANALYST</div>
      <h2>What Should We Do Next?</h2>
      <p class="sub">Prioritize the highest-risk orders only when expected value exceeds cost.</p>
      <div id="recommendation" class="notice">Calculating...</div>
    </div>
  </div>`;
  for(const id of ["capacity","cost","success","value"]){
    document.getElementById(id).addEventListener("input",updateDecision);
  }
  await updateDecision();
}

async function updateDecision() {
  const capacity=+document.getElementById("capacity").value;
  const cost=+document.getElementById("cost").value;
  const success=+document.getElementById("success").value;
  const value=+document.getElementById("value").value;

  document.getElementById("capacityLabel").textContent=capacity+"%";
  document.getElementById("costLabel").textContent="R$ "+cost;
  document.getElementById("successLabel").textContent=success+"%";
  document.getElementById("valueLabel").textContent="R$ "+value;

  const params=new URLSearchParams({
    capacity:capacity/100,cost,success:success/100,value
  });
  const d=await api("decisions?"+params);

  document.getElementById("decisionResults").innerHTML=`
    <div class="result"><small>Orders Flagged</small><strong>${fmt(d.flagged)}</strong></div>
    <div class="result"><small>Delays Potentially Prevented</small><strong>${fmt(d.expected_prevented,0)}</strong></div>
    <div class="result"><small>Intervention Cost</small><strong>${money(d.intervention_cost)}</strong></div>
    <div class="result"><small>Estimated Benefit</small><strong class="pink">${money(d.gross_benefit)}</strong></div>
    <div class="result"><small>Estimated Net Value</small><strong>${money(d.net_benefit)}</strong></div>
    <div class="result"><small>Simulated ROI</small><strong>${pct(d.roi)}</strong></div>`;

  document.getElementById("recommendation").textContent=
    d.net_benefit>0
      ? "This scenario has positive estimated value. Validate the assumptions in a randomized experiment before rollout."
      : "This scenario loses money under the selected assumptions. Reduce cost, narrow targeting or test higher-value interventions.";

  if(sceneCleanup){sceneCleanup();sceneCleanup=null;}
  scene3D("balance3d",[
    Math.max(d.intervention_cost,1),
    Math.max(d.gross_benefit,1)
  ]);
}

const pages={overview,customers,predictions,decisions};

async function navigate(page) {
  if(!pages[page])page="overview";
  clearPage();
  document.querySelectorAll("nav button").forEach(b=>{
    b.classList.toggle("active",b.dataset.page===page);
  });
  app.innerHTML='<div class="loading">Loading historical data...</div>';
  history.replaceState(null,"","#"+page);
  try {
    await pages[page]();
  } catch(error) {
    console.error(error);
    app.innerHTML=`<div class="error"><h2>Data could not load</h2>
      <p>${error.message}</p>
      <p>Check your Olist CSV files and champion audit reports.</p></div>`;
  }
}

document.querySelectorAll("nav button").forEach(b=>{
  b.addEventListener("click",()=>navigate(b.dataset.page));
});
navigate(location.hash.slice(1)||"overview");
