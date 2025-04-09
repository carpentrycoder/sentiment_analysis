import { useEffect, useState } from "react";
import axios from "axios";

const Results = () => {
  const [result, setResult] = useState(null);

  useEffect(() => {
    const fetchSessionData = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:8000/api/session-data/", {
          withCredentials: true,
        });
        console.log("✅ Fetched Data:", response.data);
        setResult(response.data);
      } catch (error) {
        console.error("❌ Fetch Error:", error);
        setResult({ error: "Failed to fetch data" });
      }
    };

    fetchSessionData();
  }, []);

  return (
    <div>
      <h2>Stored Data</h2>

      {result ? (
        <>
          <p><strong>LinkedIn:</strong> {result.linkedin || "Not available"}</p>
          <p><strong>GitHub:</strong> {result.github || "Not available"}</p>
          <p><strong>Resume Text:</strong></p>
          <pre style={{ whiteSpace: "pre-wrap" }}>{result.resume_text || "No resume text stored"}</pre>
        </>
      ) : (
        <p>Loading data...</p>
      )}
    </div>
  );
};

export default Results;
