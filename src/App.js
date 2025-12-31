import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Supervised from "./components/Supervised.js";
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  return (
    <Router> 
      <div>
        {/* Navbar + Hero sections here */}

        {/* Paradigm Cards */}
        <section className="py-5">
          <div className="container">
            <div className="row g-4">
              {/* Supervised Card */}
              <div className="col-md-6 col-lg-3">
                <Link to="/Supervised" className="text-decoration-none">
                  <div className="card border-primary h-100 text-center shadow-sm">
                    <div className="card-body">
                      <h5 className="card-title text-primary">Supervised Learning</h5>
                      <p className="card-text">Learn from labeled data to make predictions</p>
                      <p className="small text-muted">Includes: Classification, Regression, Time Series</p>
                      <p className="fw-bold text-primary">60+ algorithms</p>
                    </div>
                  </div>
                </Link>
              </div>

              {/* Other cards can remain static for now */}
            </div>
          </div>
        </section>

        {/* Routes */}
        <Routes>
          <Route path="/Supervised" element={<Supervised />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
