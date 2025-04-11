import { useState } from "react";
import axios from "axios";

const Results = () => {
  const [result, setResult] = useState(null);
  const [sendResumeChecked, setSendResumeChecked] = useState(false);
  const [consentChecked, setConsentChecked] = useState(false);

  const fetchSessionData = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:8000/api/session-data/", { withCredentials: true });
      console.log("✅ Fetched Data:", response.data);
      setResult(response.data);
    } catch (error) {
      console.error("❌ Fetch Error:", error);
      setResult({ error: "Failed to fetch data" });
    }
  };

  const handleSubmit = () => {
    // Here you can handle the form submission or refresh the page
    window.location.reload(); // Refreshes the page
  };

  return (
    <>
      {/* Blue Card */}
      <div
        style={{
          backgroundColor: "#2e3a59",
          width: "150vh",
          height: "70vh",
          overflowY: "auto",
          margin: "5vh auto",
          padding: "2rem",
          color: "white",
          borderRadius: "10px",
          boxSizing: "border-box",
        }}
      >
        <h2 style={{ fontSize: "1.5rem", marginBottom: "1rem" }}>Fetched Data</h2>

        {result ? (
          <>
            <p><strong>LinkedIn:</strong> {result.linkedin || "Not available"}</p>
            <p><strong>GitHub:</strong> {result.github || "Not available"}</p>
            <p><strong>Resume Text:</strong></p>
            <pre style={{ whiteSpace: "pre-wrap", backgroundColor: "#3b4a6b", padding: "1rem", borderRadius: "5px" }}>
              {result.resume_text || "No resume text stored"}
            </pre>
          </>
        ) : (
          <>
            <p><strong>LinkedIn:</strong> johndoe123</p>
            <p><strong>GitHub:</strong> github.com/johndoe</p>
            <p><strong>Resume Text:</strong></p>
            <pre
              style={{
                whiteSpace: "pre-wrap",
                backgroundColor: "#3b4a6b",
                padding: "1rem",
                borderRadius: "5px",
                marginBottom: "2rem",
              }}
            >
{`Experienced full-stack developer with expertise in React, Node.js, and Python. Passionate about building scalable web applications and working with cutting-edge technologies. 

- Led a team of developers in creating a real-time chat app with socket.io.
- Designed RESTful APIs using Django and Flask.
- Built reusable React components for a large-scale admin dashboard.

Skills:
- Frontend: React, Redux, Tailwind, Bootstrap
- Backend: Node.js, Express, Django, Flask
- DevOps: Docker, GitHub Actions, AWS EC2
- Database: PostgreSQL, MongoDB, Redis

Education:
B.Tech in Computer Science - XYZ University (2016 - 2020)

Certifications:
- AWS Certified Developer
- Google UX Design Certificate

Hobbies:
- Photography
- Chess
- Hiking

Additional Projects:
- Personal portfolio with dark/light toggle
- CLI tool for GitHub repo cloning and management
- Realtime collaboration whiteboard app with WebSockets
`}
            </pre>
          </>
        )}
      </div>

      {/* Checkbox Section */}
      <div
        style={{
          width: "80%",
          margin: "2rem auto",
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-start",
          gap: "1rem",
        }}
      >
        <label
          style={{
            display: "flex",
            alignItems: "center",
            fontSize: "1rem",
            color: "#2e3a59",
            backgroundColor: "#f5f5f5",
            padding: "0.75rem 1rem",
            borderRadius: "8px",
            boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
          }}
        >
          <input
            type="checkbox"
            checked={sendResumeChecked}
            onChange={() => setSendResumeChecked(!sendResumeChecked)}
            style={{ marginRight: "0.5rem" }}
          />
          If you want to send your resume to similar job recruiting companies, check this box
        </label>

        {sendResumeChecked && (
          <label
            style={{
              display: "flex",
              alignItems: "center",
              fontSize: "1rem",
              color: "#2e3a59",
              backgroundColor: "#f5f5f5",
              padding: "0.75rem 1rem",
              borderRadius: "8px",
              boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
              marginLeft: "1.25rem",
            }}
          >
            <input
              type="checkbox"
              checked={consentChecked}
              onChange={() => setConsentChecked(!consentChecked)}
              style={{ marginRight: "0.5rem" }}
            />
            If you consent the HR of the received company to evaluate your LinkedIn profile
          </label>
        )}
      </div>

      {/* Submit Button */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          marginTop: "2rem",
        }}
      >
        <button
          onClick={handleSubmit}
          style={{
            backgroundColor: "#007BFF",
            color: "white",
            padding: "0.75rem 2rem",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            fontSize: "1rem",
            boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
          }}
        >
          Submit
        </button>
      </div>
    </>
  );
};

export default Results;
