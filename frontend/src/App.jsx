import React, { useState } from "react";
import FormComponent from "./components/FormComponent";
import Results from "./components/Results"; // Import Results component
import "./App.css";

function App() {
    const [result, setResult] = useState(null); // Store fetched results

    return (
        <div>
            <FormComponent setResult={setResult} />
            <Results result={result} /> {/* Pass result to Results */}
        </div>
    );
}

export default App;