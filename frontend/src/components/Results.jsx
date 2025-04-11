import { useEffect, useState, useRef } from "react";
import axios from "axios";
import "../App.css";

const dummyText = Array(100).fill("This is sample resume data. ").join(" ");

const Results = () => {
  const [result, setResult] = useState(null);
  const [sendResumeChecked, setSendResumeChecked] = useState(false);
  const [consentChecked, setConsentChecked] = useState(false);
  const [boxHeight, setBoxHeight] = useState(250);
  const maxScrollTopRef = useRef(0);

  const resumeRef = useRef(null);

  useEffect(() => {
    const fetchSessionData = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:8000/api/market-value/", {
          withCredentials: true,
        });
        setResult(response.data);
      } catch (error) {
        console.error("Fetch error:", error);
        setResult({ error: "Failed to fetch data" });
      }
    };

    fetchSessionData();
  }, []);

  const handleScroll = () => {
    const el = resumeRef.current;
    if (!el) return;

    const scrollTop = el.scrollTop;
    const scrollHeight = el.scrollHeight - el.clientHeight;

    // Lock max scroll position
    if (scrollTop > maxScrollTopRef.current) {
      maxScrollTopRef.current = scrollTop;

      const minHeight = 250;
      const maxHeight = 500;

      const newHeight =
        minHeight + (scrollTop / scrollHeight) * (maxHeight - minHeight);

      setBoxHeight(Math.min(newHeight, maxHeight));
    }
  };

  const handleSubmit = () => {
    window.location.reload();
  };

  return (
    <div style={{ width: "90vw", maxWidth: "1400px", margin: "0 auto" }}>
      <div
        style={{
          backgroundColor: "#7b6be6",
          borderTopLeftRadius: "12px",
          borderTopRightRadius: "12px",
          padding: "2.5rem",
          color: "white",
          fontSize: "2.4rem",
          fontWeight: 700,
          textAlign: "center",
          letterSpacing: "0.5px",
          marginBottom: "-10px",
          boxShadow: "0 4px 15px rgba(0, 0, 0, 0.1)",
          position: "sticky",
          top: 0,
          zIndex: 10,
        }}
      >
        Resume Results
      </div>

      <div
        style={{
          backgroundColor: "#fff",
          padding: "2rem",
          borderBottomLeftRadius: "10px",
          borderBottomRightRadius: "10px",
          boxShadow: "0 8px 40px rgba(0, 0, 0, 0.15)",
          marginBottom: "2rem",
        }}
      >
        <h2 style={{ marginBottom: "1.5rem", color: "#333" }}>Fetched Data</h2>

        <div style={{ lineHeight: "1.7", color: "#333", fontSize: "1rem" }}>
          <p>
            <strong>LinkedIn:</strong> {result?.linkedin || "Not available"}
          </p>
          <p>
            <strong>GitHub:</strong> {result?.github || "Not available"}
          </p>
        </div>

        <div style={{ marginTop: "1rem" }}>
          <p>
            <strong>Resume Text:</strong>
          </p>
          <div
            ref={resumeRef}
            onScroll={handleScroll}
            style={{
              whiteSpace: "pre-wrap",
              backgroundColor: "#f5f5f5",
              padding: "1rem",
              borderRadius: "8px",
              height: `${boxHeight}px`,
              overflowY: "scroll",
              fontSize: "0.95rem",
              color: "#444",
              transition: "height 0.3s ease-in-out",
              boxShadow: "inset 0 0 10px rgba(0,0,0,0.05)",
              lineHeight: "1.6",
            }}
          >
            {result?.resume_text || dummyText}
          </div>
        </div>
      </div>

      <div
        style={{
          backgroundColor: "#fff",
          padding: "2rem",
          borderRadius: "10px",
          boxShadow: "0 8px 40px rgba(0, 0, 0, 0.1)",
          marginBottom: "2rem",
          display: "flex",
          flexDirection: "column",
          gap: "1rem",
        }}
      >
        <label
          style={{
            display: "flex",
            alignItems: "center",
            fontSize: "1rem",
            color: "#2e3a59",
          }}
        >
          <input
            type="checkbox"
            checked={sendResumeChecked}
            onChange={() => setSendResumeChecked(!sendResumeChecked)}
            style={{ marginRight: "0.75rem" }}
          />
          Send my resume to similar job recruiting companies
        </label>

        {sendResumeChecked && (
          <label
            style={{
              display: "flex",
              alignItems: "center",
              fontSize: "1rem",
              color: "#2e3a59",
              marginLeft: "1rem",
            }}
          >
            <input
              type="checkbox"
              checked={consentChecked}
              onChange={() => setConsentChecked(!consentChecked)}
              style={{ marginRight: "0.75rem" }}
            />
            I consent HR to evaluate my LinkedIn profile
          </label>
        )}

        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            onClick={handleSubmit}
            style={{
              padding: "0.5rem 1.25rem",
              fontSize: "0.9rem",
              backgroundColor: "#e4ddfb",
              color: "#4b4b4b",
              border: "none",
              borderRadius: "20px",
              cursor: "pointer",
              transition: "transform 0.2s ease",
            }}
            onMouseEnter={(e) => (e.target.style.transform = "scale(1.05)")}
            onMouseLeave={(e) => (e.target.style.transform = "scale(1)")}
          >
            Submit
          </button>
        </div>
      </div>
    </div>
  );
};

export default Results;