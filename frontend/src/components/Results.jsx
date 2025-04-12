import "../App.css";
import { useEffect, useState } from "react";
import axios from "axios";


const Results = () => {
  const [result, setResult] = useState(null);
  const [dummyText] = useState(Array(100).fill("This is sample resume data. ").join(" "));

  useEffect(() => {
    const storedData = localStorage.getItem("resumeData");
    if (storedData) {
      console.log("📦 Loaded from localStorage");
      setResult(JSON.parse(storedData));
    } else {
      const fetchResumeData = async () => {
        try {
          const response = await axios.get("http://127.0.0.1:8000/api/resume-parsed-info/", {
            withCredentials: true,
          });
          console.log("🌐 Fetched from API:", response.data);
          setResult(response.data);
          localStorage.setItem("resumeData", JSON.stringify(response.data)); // 🔐 Save to localStorage
        } catch (error) {
          console.error("Fetch error:", error);
          setResult({ error: "Failed to fetch data" });
        }
      };

      fetchResumeData();
    }
  }, []);

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
          <p><strong>LinkedIn:</strong> {result?.linkedin || "Not available"}</p>
          <p><strong>GitHub:</strong> {result?.github || "Not available"}</p>
        </div>

        <div style={{ marginTop: "1rem" }}>
          <p><strong>Resume Text:</strong></p>
          <div
            style={{
              whiteSpace: "pre-wrap",
              backgroundColor: "#f5f5f5",
              padding: "1rem",
              borderRadius: "8px",
              height: "300px",
              overflowY: "scroll",
              fontSize: "0.95rem",
              color: "#444",
              boxShadow: "inset 0 0 10px rgba(0,0,0,0.05)",
              lineHeight: "1.6",
            }}
          >
            {result?.resume_text || dummyText}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;
