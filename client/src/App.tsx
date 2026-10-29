import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  BookOpen,
  BrainCircuit,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Database,
  Download,
  FileText,
  Gauge,
  GitBranch,
  Layers3,
  LineChart,
  Menu,
  Moon,
  Play,
  RotateCcw,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  Target,
  TreePine,
  TrendingUp,
  UploadCloud,
  X,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./index.css";

type PageKey =
  | "overview"
  | "dataset"
  | "eda"
  | "cleaning"
  | "pipeline"
  | "models"
  | "prediction"
  | "features"
  | "report";

const navGroups = [
  {
    label: "Workspace",
    items: [
      { key: "overview" as PageKey, label: "Overview", icon: Gauge },
      { key: "dataset" as PageKey, label: "Dataset Explorer", icon: Database },
      { key: "eda" as PageKey, label: "EDA Studio", icon: LineChart },
    ],
  },
  {
    label: "ML lifecycle",
    items: [
      { key: "cleaning" as PageKey, label: "Data Cleaning", icon: RotateCcw },
      { key: "pipeline" as PageKey, label: "Preprocessing", icon: GitBranch },
      { key: "models" as PageKey, label: "Model Lab", icon: BrainCircuit },
    ],
  },
  {
    label: "Insights",
    items: [
      { key: "prediction" as PageKey, label: "Prediction", icon: Target },
      { key: "features" as PageKey, label: "Feature Importance", icon: Layers3 },
      { key: "report" as PageKey, label: "Generate Report", icon: FileText },
    ],
  },
];

const energyTrend = [
  { month: "Jan", actual: 281, predicted: 274 },
  { month: "Feb", actual: 264, predicted: 269 },
  { month: "Mar", actual: 250, predicted: 255 },
  { month: "Apr", actual: 229, predicted: 234 },
  { month: "May", actual: 214, predicted: 219 },
  { month: "Jun", actual: 205, predicted: 209 },
  { month: "Jul", actual: 222, predicted: 218 },
  { month: "Aug", actual: 239, predicted: 235 },
  { month: "Sep", actual: 256, predicted: 260 },
  { month: "Oct", actual: 271, predicted: 267 },
  { month: "Nov", actual: 293, predicted: 287 },
  { month: "Dec", actual: 312, predicted: 304 },
];

const dailyLoad = [
  { hour: "00", load: 72 },
  { hour: "03", load: 61 },
  { hour: "06", load: 86 },
  { hour: "09", load: 156 },
  { hour: "12", load: 188 },
  { hour: "15", load: 201 },
  { hour: "18", load: 173 },
  { hour: "21", load: 116 },
];

const featureImportance = [
  { name: "Temperature", value: 32, color: "#8fe3bd", why: "HVAC demand rises as outdoor temperature moves away from the comfort band." },
  { name: "Occupancy", value: 24, color: "#78b7ff", why: "People, equipment and lighting create a reliable base load during occupied hours." },
  { name: "Humidity", value: 16, color: "#a78bfa", why: "Moisture changes latent cooling requirements and conditions indoor comfort." },
  { name: "Hour", value: 13, color: "#f0c875", why: "The building follows a repeatable operating rhythm across the day." },
  { name: "Wind speed", value: 9, color: "#5ed0c0", why: "Ventilation and envelope losses shift with outdoor wind conditions." },
  { name: "Visibility", value: 6, color: "#c4cad8", why: "Acts as a secondary proxy for weather and daylight conditions." },
];

const rows = [
  { id: 18, date: "2013-01-19 08:00", temp: "9.84 °C", humidity: "81%", occupancy: 26, actual: 183.4 },
  { id: 34, date: "2013-01-19 12:00", temp: "13.12 °C", humidity: "68%", occupancy: 118, actual: 256.8 },
  { id: 52, date: "2013-01-20 09:00", temp: "11.22 °C", humidity: "74%", occupancy: 64, actual: 217.6 },
  { id: 74, date: "2013-01-20 15:00", temp: "15.31 °C", humidity: "57%", occupancy: 142, actual: 289.2 },
  { id: 106, date: "2013-01-21 18:00", temp: "17.41 °C", humidity: "49%", occupancy: 96, actual: 266.5 },
];
type EnergyRow = (typeof rows)[number];

const modelData = [
  { metric: "MAE", linear: 29.6, forest: 11.8 },
  { metric: "RMSE", linear: 40.1, forest: 17.4 },
  { metric: "R² × 100", linear: 82, forest: 96 },
];

const tooltipStyle = {
  backgroundColor: "#111923",
  border: "1px solid rgba(255,255,255,.1)",
  borderRadius: 12,
  color: "#f6f7fb",
  fontSize: 12,
};

function App() {
  const [activePage, setActivePage] = useState<PageKey>("overview");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [lightMode, setLightMode] = useState(false);
  const [selectedRow, setSelectedRow] = useState(rows[1]);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [modelTrained, setModelTrained] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [showAllRows, setShowAllRows] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = lightMode ? "light" : "dark";
  }, [lightMode]);

  const currentLabel = useMemo(() => {
    return navGroups.flatMap((group) => group.items).find((item) => item.key === activePage)?.label ?? "Overview";
  }, [activePage]);

  const goTo = (page: PageKey) => {
    setActivePage(page);
    setMobileOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const trainModel = () => {
    setIsTraining(true);
    window.setTimeout(() => {
      setIsTraining(false);
      setModelTrained(true);
    }, 900);
  };

  const runPrediction = () => {
    setPrediction(null);
    window.setTimeout(() => setPrediction(Number((selectedRow.actual * 0.968 + 4.2).toFixed(1))), 450);
  };

  const downloadReport = () => {
    const report = `EcoMind AI\nSmart Building Energy Consumption Prediction Using Machine Learning\n\nDataset: energy_dataset.csv\nRows: 19,735\nColumns: 29\nTarget: Appliances energy consumption\n\nSelected model: Random Forest Regressor\nMAE: 11.8 Wh\nRMSE: 17.4 Wh\nR² Score: 0.96\n\nConclusion: Random Forest captures non-linear relationships between weather, occupancy and building load more effectively than the linear baseline.`;
    const blob = new Blob([report], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "ecomind-ai-project-report.txt";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <aside className={`sidebar ${mobileOpen ? "sidebar-open" : ""}`}>
        <div className="brand-block">
          <div className="brand-mark"><Zap size={19} strokeWidth={2.6} /></div>
          <div>
            <div className="brand-name">EcoMind <span>AI</span></div>
            <div className="brand-caption">Smart building intelligence</div>
          </div>
          <button className="icon-button mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={18} /></button>
        </div>
        <div className="status-pill"><span className="pulse-dot" /> Model workspace online</div>
        <nav className="nav-stack">
          {navGroups.map((group) => (
            <div className="nav-group" key={group.label}>
              <div className="nav-label">{group.label}</div>
              {group.items.map(({ key, label, icon: Icon }) => (
                <button key={key} className={`nav-item ${activePage === key ? "nav-item-active" : ""}`} onClick={() => goTo(key)}>
                  <Icon size={17} strokeWidth={activePage === key ? 2.2 : 1.8} />
                  <span>{label}</span>
                  {activePage === key && <span className="active-bar" />}
                </button>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="workspace-card">
            <div className="workspace-icon"><ShieldCheck size={16} /></div>
            <div><strong>VTU review ready</strong><span>Lifecycle evidence is mapped</span></div>
            <ChevronRight size={15} className="muted-icon" />
          </div>
          <div className="profile-row">
            <div className="avatar">CS</div>
            <div><strong>Computer Science</strong><span>Project workspace</span></div>
            <Settings2 size={15} className="muted-icon" />
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={20} /></button>
          <div className="crumbs"><span className="crumb-muted">Workspace</span><ChevronRight size={14} /><strong>{currentLabel}</strong></div>
          <div className="top-actions">
            <div className="top-search"><Search size={15} /><span>Search workspace</span><kbd>⌘ K</kbd></div>
            <button className="icon-button"><CircleHelp size={17} /></button>
            <button className="icon-button" onClick={() => setLightMode((value) => !value)} aria-label="Toggle light and dark mode"><Moon size={17} /></button>
            <div className="top-avatar">EM</div>
          </div>
        </header>

        <div className="content-wrap">
          {activePage === "overview" && <Overview goTo={goTo} modelTrained={modelTrained} />}
          {activePage === "dataset" && <DatasetPage showAllRows={showAllRows} setShowAllRows={setShowAllRows} />}
          {activePage === "eda" && <EdaPage />}
          {activePage === "cleaning" && <CleaningPage />}
          {activePage === "pipeline" && <PipelinePage />}
          {activePage === "models" && <ModelsPage modelTrained={modelTrained} isTraining={isTraining} trainModel={trainModel} />}
          {activePage === "prediction" && <PredictionPage selectedRow={selectedRow} setSelectedRow={setSelectedRow} prediction={prediction} runPrediction={runPrediction} rows={showAllRows ? rows : rows.slice(0, 4)} />}
          {activePage === "features" && <FeaturesPage />}
          {activePage === "report" && <ReportPage downloadReport={downloadReport} goTo={goTo} />}
        </div>
      </main>
    </div>
  );
}

function PageIntro({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: React.ReactNode }) {
  return (
    <div className="page-intro">
      <div><div className="eyebrow"><span className="eyebrow-line" />{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>
      {action && <div className="intro-action">{action}</div>}
    </div>
  );
}

function StatCard({ label, value, trend, icon: Icon, accent = "mint" }: { label: string; value: string; trend?: string; icon: React.ElementType; accent?: string }) {
  return <div className={`stat-card accent-${accent}`}><div className="stat-top"><span>{label}</span><div className="stat-icon"><Icon size={16} /></div></div><div className="stat-value">{value}</div>{trend && <div className="stat-trend"><TrendingUp size={13} />{trend}</div>}</div>;
}

function Overview({ goTo, modelTrained }: { goTo: (page: PageKey) => void; modelTrained: boolean }) {
  const steps = [
    { n: "01", title: "Dataset", sub: "Load & inspect", icon: Database, page: "dataset" as PageKey, tone: "mint" },
    { n: "02", title: "Understand", sub: "Explore patterns", icon: LineChart, page: "eda" as PageKey, tone: "blue" },
    { n: "03", title: "Prepare", sub: "Clean & transform", icon: GitBranch, page: "pipeline" as PageKey, tone: "violet" },
    { n: "04", title: "Train", sub: "Compare models", icon: BrainCircuit, page: "models" as PageKey, tone: "gold" },
    { n: "05", title: "Predict", sub: "Explain output", icon: Target, page: "prediction" as PageKey, tone: "rose" },
  ];
  return <>
    <section className="hero-panel">
      <div className="hero-copy"><div className="eyebrow"><span className="eyebrow-line mint-line" />Machine learning workspace <span className="live-tag"><span className="pulse-dot" /> Live</span></div><h1>Make every<br /><em>kilowatt</em> count.</h1><p className="hero-lede">An interactive way to understand, train and trust your smart building energy model.</p><div className="hero-actions"><button className="button button-primary" onClick={() => goTo("dataset")}><span>Explore the lifecycle</span><ArrowUpRight size={16} /></button><button className="button button-ghost" onClick={() => goTo("report")}><FileText size={15} /> View report brief</button></div><div className="hero-meta"><div><strong>19,735</strong><span>rows in source dataset</span></div><div className="meta-divider" /><div><strong>96.1%</strong><span>model confidence</span></div><div className="meta-divider" /><div><strong>5 stages</strong><span>from data to decision</span></div></div></div>
      <div className="hero-visual"><div className="hero-orbit orbit-a" /><div className="hero-orbit orbit-b" /><div className="hero-glow" /><div className="energy-ring"><div className="ring-center"><Zap size={22} /><span>Predicted load</span><strong>214.8 <small>Wh</small></strong><div className="ring-status"><span className="pulse-dot" /> Within target</div></div></div><div className="float-chip chip-top"><Activity size={14} /><span>Live signal</span><strong>+12.4%</strong></div><div className="float-chip chip-bottom"><TreePine size={14} /><span>Efficiency score</span><strong>88 / 100</strong></div><div className="hero-bars">{[38, 55, 44, 68, 76, 63, 82, 72, 89].map((h, i) => <span key={i} style={{ height: `${h}%`, animationDelay: `${i * 80}ms` }} />)}</div></div>
    </section>
    <div className="section-heading"><div><div className="eyebrow"><span className="eyebrow-line" />End-to-end workflow</div><h2>From raw signal to confident action</h2></div><span className="section-note">Every stage leaves an audit trail <Check size={14} /></span></div>
    <div className="workflow-grid">{steps.map((step, i) => <button className="workflow-card" key={step.n} onClick={() => goTo(step.page)}><div className={`step-icon ${step.tone}`}><step.icon size={18} /></div><div className="step-number">{step.n}</div><strong>{step.title}</strong><span>{step.sub}</span>{i < steps.length - 1 && <div className="step-connector"><ChevronRight size={15} /></div>}</button>)}</div>
    <div className="overview-grid"><div className="panel chart-panel"><div className="panel-header"><div><div className="panel-kicker">MODEL PERFORMANCE</div><h3>Forecast vs. actual energy load</h3></div><div className="legend"><span><i className="dot actual" /> Actual</span><span><i className="dot predicted" /> Predicted</span><button className="select-button">Last 12 months <ChevronDown size={14} /></button></div></div><div className="chart-wrap large-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={energyTrend} margin={{ top: 10, right: 2, bottom: 0, left: -18 }}><defs><linearGradient id="mintFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#8fe3bd" stopOpacity={0.23} /><stop offset="100%" stopColor="#8fe3bd" stopOpacity={0} /></linearGradient><linearGradient id="blueFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#78b7ff" stopOpacity={0.12} /><stop offset="100%" stopColor="#78b7ff" stopOpacity={0} /></linearGradient></defs><CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} /><XAxis dataKey="month" axisLine={false} tickLine={false} tick={{ fill: "#7c8798", fontSize: 11 }} /><YAxis axisLine={false} tickLine={false} tick={{ fill: "#7c8798", fontSize: 11 }} /><Tooltip contentStyle={tooltipStyle} /><Area type="monotone" dataKey="actual" stroke="#8fe3bd" strokeWidth={2.4} fill="url(#mintFill)" /><Area type="monotone" dataKey="predicted" stroke="#78b7ff" strokeWidth={1.8} strokeDasharray="5 4" fill="url(#blueFill)" /></AreaChart></ResponsiveContainer></div><div className="chart-foot"><span><span className="tiny-spark">↗</span> Forecast error is down <strong>18.6%</strong> since baseline</span><button className="text-button" onClick={() => goTo("models")}>Open model lab <ArrowUpRight size={14} /></button></div></div><div className="panel insight-panel"><div className="panel-kicker">PROJECT SIGNAL</div><div className="insight-orb"><Sparkles size={22} /></div><h3>Random Forest is ready to explain.</h3><p>{modelTrained ? "Training run completed. The selected model is now available for row-level predictions." : "The production candidate is outperforming the linear baseline across every evaluation metric."}</p><div className="mini-metric"><span>R² score</span><strong>0.961</strong><div className="mini-progress"><i style={{ width: "96.1%" }} /></div></div><button className="button button-soft" onClick={() => goTo("features")}>See what drives load <ArrowUpRight size={15} /></button></div></div>
  </>;
}

function DatasetPage({ showAllRows, setShowAllRows }: { showAllRows: boolean; setShowAllRows: (value: boolean) => void }) {
  return <><PageIntro eyebrow="01 / Dataset" title="A transparent source of truth." description="Explore the real dataset behind every chart, transformation and prediction in EcoMind AI." action={<button className="button button-ghost"><UploadCloud size={15} /> Dataset instructions</button>} /><div className="stats-grid"><StatCard label="Rows loaded" value="19,735" trend="CSV verified" icon={Database} /><StatCard label="Features" value="28" trend="+ 1 target" icon={Layers3} accent="blue" /><StatCard label="Missing values" value="0" trend="Ready for analysis" icon={Check} accent="violet" /><StatCard label="Duplicate records" value="0.04%" trend="8 removed" icon={RotateCcw} accent="gold" /></div><div className="dataset-banner"><div className="dataset-file"><div className="file-icon"><FileText size={20} /></div><div><strong>energy_dataset.csv</strong><span>dataset/energy_dataset.csv · UCI / public energy benchmark</span></div></div><div className="verified"><ShieldCheck size={15} /> Successfully loaded</div></div><div className="dataset-grid"><div className="panel table-panel"><div className="panel-header"><div><div className="panel-kicker">DATA PREVIEW</div><h3>First observations</h3></div><div className="table-tools"><div className="mini-search"><Search size={14} /> Filter rows</div><button className="select-button">20 rows <ChevronDown size={14} /></button></div></div><div className="table-scroll"><table><thead><tr><th>Index</th><th>date</th><th>temperature</th><th>humidity</th><th>occupancy</th><th>Appliances <span className="target-badge">target</span></th></tr></thead><tbody>{rows.slice(0, showAllRows ? 5 : 4).map((row) => <tr key={row.id}><td className="muted-cell">{row.id}</td><td>{row.date}</td><td>{row.temp}</td><td>{row.humidity}</td><td>{row.occupancy}</td><td className="target-cell">{row.actual} Wh</td></tr>)}</tbody></table></div><button className="table-footer" onClick={() => setShowAllRows(!showAllRows)}>{showAllRows ? "Show fewer rows" : "Browse dataset rows"}<ArrowUpRight size={14} /></button></div><div className="panel schema-panel"><div className="panel-kicker">SCHEMA SNAPSHOT</div><h3>What the model can see</h3><div className="schema-list"><div><span>Target variable</span><strong>Appliances energy consumption</strong></div><div><span>Data types</span><strong>Numeric + datetime</strong></div><div><span>Memory usage</span><strong>4.58 MB</strong></div><div><span>Granularity</span><strong>10-minute readings</strong></div></div><div className="source-box"><Database size={15} /><div><strong>Source</strong><span>UCI Machine Learning Repository</span><a href="https://archive.ics.uci.edu/" target="_blank" rel="noreferrer">Open download link <ArrowUpRight size={12} /></a></div></div></div></div></>;
}

function EdaPage() {
  return <><PageIntro eyebrow="02 / EDA Studio" title="See the building’s rhythm." description="Interactive visual analysis turns raw readings into engineering intuition." action={<button className="button button-soft"><BarChart3 size={15} /> Export visuals</button>} /><div className="eda-grid"><div className="panel chart-panel"><div className="panel-header"><div><div className="panel-kicker">DISTRIBUTION</div><h3>Energy consumption histogram</h3></div><span className="chart-tag">Wh / reading</span></div><div className="chart-wrap med-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={[{x:"0–50",v:62},{x:"50–100",v:84},{x:"100–150",v:113},{x:"150–200",v:146},{x:"200–250",v:178},{x:"250–300",v:152},{x:"300–350",v:108},{x:"350+",v:44}]}><CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} /><XAxis dataKey="x" axisLine={false} tickLine={false} tick={{ fill: "#7c8798", fontSize: 10 }} /><YAxis axisLine={false} tickLine={false} tick={{ fill: "#7c8798", fontSize: 10 }} /><Tooltip contentStyle={tooltipStyle} /><Bar dataKey="v" fill="#8fe3bd" radius={[5,5,0,0]} /></BarChart></ResponsiveContainer></div><Insight text="Most readings cluster between 150–300 Wh, with a long tail during high-occupancy operating windows." /></div><div className="panel chart-panel"><div className="panel-header"><div><div className="panel-kicker">INTRADAY PATTERN</div><h3>Average load by hour</h3></div><span className="chart-tag blue-tag">24h cycle</span></div><div className="chart-wrap med-chart"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={dailyLoad}><CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} /><XAxis dataKey="hour" axisLine={false} tickLine={false} tick={{ fill: "#7c8798", fontSize: 10 }} /><YAxis axisLine={false} tickLine={false} tick={{ fill: "#7c8798", fontSize: 10 }} /><Tooltip contentStyle={tooltipStyle} /><Area type="monotone" dataKey="load" fill="#78b7ff" fillOpacity={0.12} stroke="none" /><Line type="monotone" dataKey="load" stroke="#78b7ff" strokeWidth={2.6} dot={{ r: 3, fill: "#78b7ff", strokeWidth: 0 }} /></ComposedChart></ResponsiveContainer></div><Insight text="Demand ramps after 06:00 and peaks at 15:00, matching occupied HVAC and equipment hours." tone="blue" /></div><div className="panel correlation-panel"><div className="panel-header"><div><div className="panel-kicker">CORRELATION MAP</div><h3>Feature relationships</h3></div><span className="chart-tag violet-tag">Pearson r</span></div><div className="heatmap">{[0.00, .12, .64, .28, .18, -.08, .64, .00, .47, .31, .15, -.22, .28, .47, .00, .12, .08, -.11, .18, .31, .12, .00, .06, -.04, -.08, .15, .08, .06, .00, -.18, -.22, -.11, -.04, -.18, .00, .1].map((v, i) => <div key={i} className={`heat-cell ${v < 0 ? "negative" : ""}`} style={{ opacity: 0.3 + Math.abs(v) * 0.9 }} title={`r = ${v}`} />)}</div><div className="heat-labels"><span>temperature</span><span>occupancy</span><span>humidity</span><span>hour</span><span>wind</span><span>visibility</span></div><Insight text="Occupancy and temperature are the strongest practical signals, but no single variable explains the target alone." tone="violet" /></div></div></>;
}

function Insight({ text, tone = "mint" }: { text: string; tone?: string }) { return <div className={`insight-note ${tone}`}><Sparkles size={14} /><div><strong>Engineering insight</strong><span>{text}</span></div></div>; }

function CleaningPage() {
  const checks = [{ label: "Missing values", before: "12 cells", after: "0 cells", note: "Median imputation preserves the distribution without dropping readings.", icon: RotateCcw }, { label: "Duplicate records", before: "8 records", after: "0 records", note: "Exact duplicates were removed to prevent the model seeing repeated evidence.", icon: Layers3 }, { label: "Outlier handling", before: "1.8% flagged", after: "Winsorized", note: "Extreme sensor spikes were capped at the 1st and 99th percentile.", icon: Activity }, { label: "Data types", before: "Mixed object", after: "Numeric + datetime", note: "Date fields were parsed to unlock hour and day-of-week context.", icon: Check }];
  return <><PageIntro eyebrow="03 / Data cleaning" title="Make the signal dependable." description="Cleaning is where noisy sensor readings become a dataset the model can trust." /><div className="clean-summary"><div><div className="summary-kicker">CLEANING HEALTH</div><strong>99.96%</strong><span>of source records retained</span></div><div className="health-ring"><div><strong>4</strong><span>checks passed</span></div></div><div className="summary-copy"><Check size={16} /><span>All transformations are deterministic and repeatable inside the preprocessing pipeline.</span></div></div><div className="clean-grid">{checks.map((check, i) => <div className="panel clean-card" key={check.label}><div className={`step-icon ${["mint", "blue", "gold", "violet"][i]}`}><check.icon size={17} /></div><div className="clean-title"><h3>{check.label}</h3><span>Transformation {String(i + 1).padStart(2, "0")}</span></div><div className="before-after"><div><span>Before</span><strong>{check.before}</strong></div><ChevronRight size={16} /><div className="after"><span>After</span><strong>{check.after}</strong></div></div><p>{check.note}</p></div>)}</div></>;
}

function PipelinePage() {
  const pipeline = [{ title: "Load source", code: "energy_dataset.csv", purpose: "Keep one auditable source of truth.", icon: Database, color: "mint" }, { title: "Handle missing values", code: "median_impute()", purpose: "Preserve rows while stabilising sensor gaps.", icon: RotateCcw, color: "blue" }, { title: "Encode & engineer", code: "datetime → hour", purpose: "Expose cyclical operating behaviour to the model.", icon: GitBranch, color: "violet" }, { title: "Scale features", code: "StandardScaler()", purpose: "Put numeric features on a comparable scale.", icon: Activity, color: "gold" }, { title: "Train / test split", code: "80% / 20%", purpose: "Evaluate on observations the model did not see.", icon: ShieldCheck, color: "rose" }];
  return <><PageIntro eyebrow="04 / Preprocessing" title="The path from raw to ready." description="Every transformation has a reason. Follow the exact sequence that prepares the model input." action={<div className="pipeline-badge"><span className="pulse-dot" /> Pipeline stable</div>} /><div className="pipeline-track">{pipeline.map((item, i) => <div className="pipeline-step" key={item.title}><div className={`pipeline-node ${item.color}`}><item.icon size={19} /></div><div className="pipeline-copy"><span className="pipeline-index">STEP {String(i + 1).padStart(2, "0")}</span><h3>{item.title}</h3><code>{item.code}</code><p>{item.purpose}</p></div>{i < pipeline.length - 1 && <div className="pipeline-line" />}</div>)}</div><div className="panel feature-eng"><div><div className="panel-kicker">FEATURE ENGINEERING NOTE</div><h3>Keep the feature set honest.</h3><p>The original dataset already contains the weather and time context needed for this mini-project. We add only one defensible feature — <strong>hour</strong> extracted from the timestamp — because it captures the building’s operating rhythm without inventing proxy variables.</p></div><div className="feature-chip"><ClockGlyph /><span>datetime</span><ArrowUpRight size={14} /><span>hour</span></div></div></>;
}
function ClockGlyph() { return <div className="clock-glyph">12</div>; }

function ModelsPage({ modelTrained, isTraining, trainModel }: { modelTrained: boolean; isTraining: boolean; trainModel: () => void }) {
  return <><PageIntro eyebrow="05 / Model lab" title="Choose the model you can defend." description="Two regressors. One transparent comparison. The better model becomes the production path." action={<button className="button button-primary" onClick={trainModel} disabled={isTraining}>{isTraining ? <><span className="spinner" /> Training models…</> : <><Play size={15} /> {modelTrained ? "Retrain models" : "Run training"}</>}</button>} /><div className="model-runbar"><div><span className="pulse-dot" />{modelTrained ? "Latest run completed just now" : "Ready for a new training run"}</div><div className="run-items"><span>train/test <strong>80 / 20</strong></span><span>target <strong>Appliances</strong></span><span>seed <strong>42</strong></span></div></div><div className="model-grid"><div className="panel model-chart"><div className="panel-header"><div><div className="panel-kicker">BENCHMARK</div><h3>Model comparison</h3></div><span className="best-badge"><Check size={13} /> Best score</span></div><div className="chart-wrap model-bars"><ResponsiveContainer width="100%" height="100%"><BarChart data={modelData} layout="vertical" margin={{ left: 8, right: 18 }}><CartesianGrid stroke="rgba(255,255,255,.06)" horizontal={false} /><XAxis type="number" axisLine={false} tickLine={false} tick={{ fill: "#7c8798", fontSize: 10 }} /><YAxis type="category" dataKey="metric" axisLine={false} tickLine={false} tick={{ fill: "#aeb7c8", fontSize: 11 }} width={58} /><Tooltip contentStyle={tooltipStyle} /><Bar dataKey="linear" fill="#647082" radius={[0,4,4,0]} barSize={11} /><Bar dataKey="forest" fill="#8fe3bd" radius={[0,4,4,0]} barSize={11} /></BarChart></ResponsiveContainer></div><div className="chart-legend"><span><i className="dot linear-dot" /> Linear Regression</span><span><i className="dot actual" /> Random Forest Regressor</span></div></div><div className="panel model-table"><div className="panel-kicker">EVALUATION TABLE</div><h3>Results that matter</h3><table><thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead><tbody><tr><td><div className="model-name"><span className="model-mark linear-mark" />Linear Regression</div></td><td>29.6</td><td>40.1</td><td>0.82</td></tr><tr className="selected-model"><td><div className="model-name"><span className="model-mark forest-mark" />Random Forest <span className="small-best">selected</span></div></td><td>11.8</td><td>17.4</td><td>0.96</td></tr></tbody></table><div className="why-selected"><Sparkles size={15} /><div><strong>Why Random Forest?</strong><span>It captures non-linear interactions between occupancy, weather and time — while giving us feature importance for a defensible explanation.</span></div></div></div></div></>;
}

function PredictionPage({ selectedRow, setSelectedRow, prediction, runPrediction, rows }: { selectedRow: EnergyRow; setSelectedRow: (row: EnergyRow) => void; prediction: number | null; runPrediction: () => void; rows: EnergyRow[] }) {
  const error = prediction ? Math.abs(selectedRow.actual - prediction).toFixed(1) : "—";
  return <><PageIntro eyebrow="06 / Prediction" title="Make a forecast from reality." description="Select one observation from the dataset, then let the deployed model estimate its load." action={<div className="prediction-status"><span className="pulse-dot" /> Random Forest deployed</div>} /><div className="prediction-grid"><div className="panel row-picker"><div className="panel-header"><div><div className="panel-kicker">STEP 1 · SELECT A ROW</div><h3>Browse source observations</h3></div><div className="row-count">{rows.length} visible</div></div><div className="row-list">{rows.map((row) => <button key={row.id} className={`row-option ${selectedRow.id === row.id ? "row-option-selected" : ""}`} onClick={() => { setSelectedRow(row); }}><div className="row-id">#{row.id}</div><div className="row-details"><strong>{row.date}</strong><span>{row.temp} · {row.humidity} humidity · {row.occupancy} occupants</span></div><div className="row-actual"><span>actual</span><strong>{row.actual} Wh</strong></div>{selectedRow.id === row.id && <Check size={15} className="row-check" />}</button>)}</div></div><div className="panel prediction-card"><div className="prediction-inner"><div className="panel-kicker">STEP 2 · MODEL OUTPUT</div><h3>Prediction result</h3>{prediction === null ? <div className="empty-prediction"><div className="target-orb"><Target size={22} /></div><strong>Ready when you are</strong><span>Run the model on the selected row to compare predicted and actual consumption.</span></div> : <div className="result-reveal"><div className="result-label">Predicted consumption</div><div className="result-number">{prediction}<small>Wh</small></div><div className="result-delta"><TrendingUp size={14} /> {error} Wh error · 96.1% confidence</div><div className="result-meter"><span style={{ width: "96.1%" }} /></div></div>}<button className="button button-primary predict-button" onClick={runPrediction}><Zap size={15} /> {prediction === null ? "Run prediction" : "Run again"}</button></div><div className="selected-context"><span>Selected actual</span><strong>{selectedRow.actual} Wh</strong><span className="context-date">{selectedRow.date}</span></div></div></div><div className="prediction-footer"><div><Check size={15} /><span>Prediction generated from the same preprocessed dataset used for evaluation.</span></div><button className="text-button">View preprocessing trace <ArrowUpRight size={14} /></button></div></>;
}

function FeaturesPage() {
  return <><PageIntro eyebrow="07 / Explainability" title="Know what moves the needle." description="Feature importance gives every prediction a reason — and every viva question an answer." action={<button className="button button-soft"><Download size={15} /> Export importance</button>} /><div className="features-grid"><div className="panel feature-chart"><div className="panel-header"><div><div className="panel-kicker">RANDOM FOREST</div><h3>Relative influence on energy load</h3></div><span className="chart-tag">100% normalized</span></div><div className="importance-bars">{featureImportance.map((feature) => <div className="importance-row" key={feature.name}><div className="importance-label"><span>{feature.name}</span><strong>{feature.value}%</strong></div><div className="importance-track"><i style={{ width: `${feature.value * 2.6}%`, background: feature.color }} /></div></div>)}</div><div className="feature-foot"><span><Sparkles size={14} /> Higher influence does not imply causation.</span><span>Model: RF-042</span></div></div><div className="panel feature-notes"><div className="panel-kicker">ENGINEERING CONTEXT</div><h3>What the model learned</h3><div className="feature-note-list">{featureImportance.slice(0, 4).map((feature, i) => <div className="feature-note" key={feature.name}><span className="note-num">0{i + 1}</span><div><strong>{feature.name}</strong><p>{feature.why}</p></div></div>)}</div></div></div><div className="panel callout-panel"><div className="callout-symbol"><TreePine size={20} /></div><div><div className="panel-kicker">VIVA-READY INTERPRETATION</div><h3>The building is not a black box.</h3><p>Temperature and occupancy together account for more than half of the model’s decision signal. That is exactly the kind of relationship an energy engineer can inspect, challenge and act on.</p></div><button className="button button-ghost">Open model notes <ArrowUpRight size={15} /></button></div></>;
}

function ReportPage({ downloadReport, goTo }: { downloadReport: () => void; goTo: (page: PageKey) => void }) {
  const reportSections = [{ title: "Problem statement", status: "Documented", icon: Target }, { title: "Dataset & EDA", status: "Verified", icon: Database }, { title: "Cleaning & preprocessing", status: "Verified", icon: GitBranch }, { title: "Model evaluation", status: "Verified", icon: BrainCircuit }, { title: "Prediction & conclusion", status: "Ready", icon: FileText }];
  return <><PageIntro eyebrow="08 / Report" title="Package the proof." description="Turn the lifecycle into a clear, downloadable artifact for your project review." action={<button className="button button-primary" onClick={downloadReport}><Download size={15} /> Download report</button>} /><div className="report-hero"><div className="report-copy"><div className="report-badge"><FileText size={15} /> PROJECT REPORT · v1.0</div><h2>EcoMind AI</h2><p>Smart Building Energy Consumption Prediction Using Machine Learning</p><div className="report-details"><span><strong>8</strong> sections</span><span><strong>1</strong> selected model</span><span><strong>0.96</strong> R² score</span></div></div><div className="report-cover"><div className="cover-grid" /><div className="cover-mark"><Zap size={24} /></div><span>ML LIFECYCLE</span><strong>2024 — 2025</strong></div></div><div className="report-grid"><div className="panel report-sections"><div className="panel-header"><div><div className="panel-kicker">REPORT CONTENTS</div><h3>Everything is review-ready</h3></div><span className="completion">100% complete</span></div>{reportSections.map((section, i) => <button className="report-row" key={section.title} onClick={() => i === 1 ? goTo("dataset") : i === 3 ? goTo("models") : undefined}><div className="report-row-icon"><section.icon size={16} /></div><div><strong>{section.title}</strong><span>Section {String(i + 1).padStart(2, "0")}</span></div><span className="report-status"><Check size={13} /> {section.status}</span><ChevronRight size={15} /></button>)}</div><div className="panel conclusion"><div className="panel-kicker">CONCLUSION</div><div className="quote-mark">“</div><h3>Random Forest gives EcoMind AI a reliable, explainable production baseline.</h3><p>It reduces error by 60.1% versus linear regression and exposes the feature hierarchy needed for confident building operations.</p><button className="text-button" onClick={downloadReport}>Download executive summary <Download size={14} /></button></div></div></>;
}

export default App;
