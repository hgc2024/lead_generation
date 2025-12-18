import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Mail, CheckCircle, AlertCircle, TrendingUp, Users } from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [leads, setLeads] = useState([]);
  const [featureImportance, setFeatureImportance] = useState({});
  const [loading, setLoading] = useState(false);
  const [selectedLead, setSelectedLead] = useState(null);
  const [generatedEmail, setGeneratedEmail] = useState(null);
  const [roiMetrics, setRoiMetrics] = useState(null);

  useEffect(() => {
    // Initial data load - assuming model is trained or we trigger training on load
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      // 1. Train/Get Model Metrics
      const trainRes = await axios.post(`${API_BASE}/train`);
      setFeatureImportance(trainRes.data.feature_importance);

      // 2. Get Leads
      const leadsRes = await axios.get(`${API_BASE}/leads`);
      setLeads(leadsRes.data);

      // 3. Get ROI
      const roiRes = await axios.get(`${API_BASE}/roi`);
      setRoiMetrics(roiRes.data);

    } catch (error) {
      console.error("Error fetching data", error);
      alert("Backend API not reachable. Make sure 'main.py' is running.");
    } finally {
      setLoading(false);
    }
  };

  const generateEmail = async (lead) => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/generate-email`, {
        lead_profile: {
          LeadId: lead['Prospect ID'] || 'Unknown',
          LeadOrigin: lead['Lead Origin'],
          LeadSource: lead['Lead Source'],
          TotalTimeSpentOnWebsite: lead['Total Time Spent on Website'],
          LastActivity: lead['Last Activity'],
          Tags: lead['Tags'] || '',
          ConvertedProbability: lead['ConvertedProbability'],
          City: lead['City']
        }
      });
      setGeneratedEmail(res.data);
      setSelectedLead(lead);
    } catch (error) {
      console.error("Error generating email", error);
      alert("Failed to generate email. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>Hybrid AI Sales Agent</h1>
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          {roiMetrics && (
            <div className="badge badge-med">
              Est. Savings: {roiMetrics.savings_per_thousand_emails}/1k emails
            </div>
          )}
          <button onClick={fetchData} disabled={loading}>
            {loading ? 'Processing...' : 'Refresh Data'}
          </button>
        </div>
      </header>

      <div className="card-grid">

        {/* Analytics Section */}
        <section className="card">
          <h2><TrendingUp size={20} style={{ marginRight: '10px' }} />Top Drivers of Conversion</h2>
          <div style={{ padding: '20px 0' }}>
            {Object.entries(featureImportance).map(([feature, score]) => (
              <div key={feature} className="bar-container">
                <span className="bar-label">{feature}</span>
                <div className="bar-fill-wrapper">
                  <div className="bar-fill" style={{ width: `${score * 100 * 2}%` }}></div> {/* Scaling for visibility */}
                </div>
                <span className="bar-value">{(score * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </section>

        {/* Lead List Section */}
        <section className="card">
          <h2><Users size={20} style={{ marginRight: '10px' }} />High Value Leads</h2>
          <div style={{ overflowX: 'auto' }}>
            <table className="leads-table">
              <thead>
                <tr>
                  <th>Lead Source</th>
                  <th>Time on Site</th>
                  <th>Tags</th>
                  <th>Conversion Probability</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {leads
                  .sort((a, b) => b.ConvertedProbability - a.ConvertedProbability)
                  .slice(0, 10) // Show top 10
                  .map((lead, idx) => (
                    <tr key={idx}>
                      <td>{lead['Lead Source']}</td>
                      <td>{lead['Total Time Spent on Website']}s</td>
                      <td>{lead['Tags']}</td>
                      <td>
                        <span className={`badge ${lead.ConvertedProbability > 0.7 ? 'badge-high' : 'badge-med'}`}>
                          {(lead.ConvertedProbability * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td>
                        <button onClick={() => generateEmail(lead)} style={{ fontSize: '0.8rem', padding: '4px 8px' }}>
                          <Mail size={14} style={{ marginRight: '5px' }} /> Draft Email
                        </button>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      {/* Email Modal */}
      {selectedLead && generatedEmail && (
        <div className="email-modal" onClick={() => setSelectedLead(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="close-btn" onClick={() => setSelectedLead(null)}>&times;</button>
            <h2>Draft for {selectedLead['Lead Source']} Lead</h2>
            <div style={{ marginBottom: '20px', color: '#ccc' }}>
              <strong>Recommended Product:</strong> {generatedEmail.product_recommended}
            </div>
            <h3>Generated Email Content:</h3>
            <div className="generated-email">
              {generatedEmail.email_content}
            </div>
            <div style={{ marginTop: '20px', display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button onClick={() => { alert("Email Sent (Simulated)!"); setSelectedLead(null); }}>
                <CheckCircle size={16} style={{ marginRight: '5px' }} /> Send Email
              </button>
              <button style={{ backgroundColor: '#2a2a47' }} onClick={() => setSelectedLead(null)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
