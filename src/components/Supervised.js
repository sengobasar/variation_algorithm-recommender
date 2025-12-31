import React from 'react';
import { Link } from 'react-router-dom';

function Supervised() {
  return (
    <div className="container py-5">
      {/* Breadcrumb */}
      <nav aria-label="breadcrumb">
        <ol className="breadcrumb">
          <li className="breadcrumb-item"><Link to="/">Home</Link></li>
          <li className="breadcrumb-item active" aria-current="page">Supervised Learning</li>
        </ol>
      </nav>

      <h2 className="mb-4">Supervised Learning</h2>

      {/* Subcategory Grid */}
      <div className="row g-4">
        <div className="col-md-4">
          <div className="card h-100 shadow-sm text-center">
            <div className="card-body">
              <h5 className="card-title">Classification</h5>
              <ul className="list-unstyled">
                <li>Binary Classification</li>
                <li>Multi-class Classification</li>
                <li>Multi-label Classification</li>
                <li>Imbalanced Classification</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="col-md-4">
          <div className="card h-100 shadow-sm text-center">
            <div className="card-body">
              <h5 className="card-title">Regression</h5>
              <ul className="list-unstyled">
                <li>Linear Regression</li>
                <li>Non-linear Regression</li>
                <li>Time Series Forecasting</li>
                <li>Multi-output Regression</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="col-md-4">
          <div className="card h-100 shadow-sm text-center">
            <div className="card-body">
              <h5 className="card-title">Probabilistic Models</h5>
              <ul className="list-unstyled">
                <li>Naive Bayes</li>
                <li>Bayesian Networks</li>
                <li>Gaussian Processes</li>
                <li>Hidden Markov Models</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Supervised;
