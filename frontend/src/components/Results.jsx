import { useEffect, useState, useRef } from "react";
import axios from "axios";
import "../App.css";

const dummyText = Array(100).fill("This is sample resume data. ").join(" ");

const Results = () => {
  const [result, setResult] = useState(null);
  const [resumeInfo, setResumeInfo] = useState(null); // Add state for resume info
  const [sendResumeChecked, setSendResumeChecked] = useState(false);
  const [consentChecked, setConsentChecked] = useState(false);
  const [boxHeight, setBoxHeight] = useState(250);
  const maxScrollTopRef = useRef(0);
  const resumeRef = useRef(null);

  useEffect(() => {
    let isMounted = true;

    const fetchData = async () => {
      try {
        // Fetch both session data and resume parsed info
        const [sessionResponse, resumeInfoResponse] = await Promise.all([
          axios.get("http://127.0.0.1:8000/api/session-data/", {
            withCredentials: true,
          }),
          axios.get("http://127.0.0.1:8000/api/resume-parsed-info/", {
            withCredentials: true,
          })
        ]);
        
        if (isMounted) {
          setResult(sessionResponse.data);
          setResumeInfo(resumeInfoResponse.data);
          console.log("Session data:", sessionResponse.data);
          console.log("Resume parsed info:", resumeInfoResponse.data);
        }
      } catch (error) {
        console.error("Fetch error:", error);
        setResult({ error: "Failed to fetch data" });
      }
    };

    fetchData();

    return () => {
      isMounted = false;
    };
  }, []);

  // Rest of your component remains the same...
  
  // Then in your JSX, you can display the resume info
  return (
    <div style={{ width: "90vw", maxWidth: "1400px", margin: "0 auto" }}>
      {/* Header and other sections remain the same */}
      
      {/* Add a new section to display the parsed resume info */}
      <section
        style={{
          backgroundColor: "#fff",
          padding: "2rem",
          borderRadius: "10px",
          boxShadow: "0 8px 40px rgba(0, 0, 0, 0.1)",
          marginBottom: "2rem",
        }}
      >
        <h2 style={{ marginBottom: "1.5rem", color: "#333" }}>Parsed Resume Information</h2>
        
        {resumeInfo ? (
          <pre style={{ 
            whiteSpace: "pre-wrap", 
            backgroundColor: "#f5f5f5",
            padding: "1rem",
            borderRadius: "8px",
            maxHeight: "500px",
            overflowY: "auto"
          }}>
            {JSON.stringify(resumeInfo, null, 2)}
          </pre>
        ) : (
          <p>Loading parsed resume information...</p>
        )}
      </section>
      
      {/* Rest of your sections remain the same */}
    </div>
  );
};

export default Results;