import { NavLink, Route, Routes } from "react-router-dom";
import DatasetsPage from "./pages/Datasets";
import DatasetAnnotationsPage from "./pages/DatasetAnnotations";
import ExperimentsPage from "./pages/Experiments";
import InferencePage from "./pages/Inference";
import ResourcesPage from "./pages/Resources";
import TrainingPage from "./pages/Training";

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>YoloFlow</h1>
        <nav>
          <NavLink to="/">Inference</NavLink>
          <NavLink to="/datasets">Datasets</NavLink>
          <NavLink to="/training">Training</NavLink>
          <NavLink to="/experiments">Experiments</NavLink>
          <NavLink to="/resources">Resources</NavLink>
        </nav>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<InferencePage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/datasets/:id/annotations" element={<DatasetAnnotationsPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/resources" element={<ResourcesPage />} />
        </Routes>
      </main>
    </div>
  );
}