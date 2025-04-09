import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const FormComponent = ({ setResult }) => {
    const [formData, setFormData] = useState({
        name: "",
        email: "",
        contact: "",
        github: "",
        linkedin: "",
        resume: null,
    });

    const navigate = useNavigate();

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData((prev) => ({ ...prev, [name]: value }));
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        setFormData((prev) => ({ ...prev, resume: file }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        const data = new FormData();
        Object.entries(formData).forEach(([key, value]) => {
            if (value) data.append(key, value);
        });

        try {
            const response = await axios.post(
                "http://127.0.0.1:8000/api/data/",
                data,
                {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                    withCredentials: true, // ✅ Send cookies to Django backend
                }
            );

            console.log("✅ Submitted Successfully:", response.data);

            // Optional delay to ensure session writes are completed
            await new Promise((resolve) => setTimeout(resolve, 300));

            const sessionRes = await axios.get(
                "http://127.0.0.1:8000/api/session-data/",
                {
                    withCredentials: true, // ✅ Also send cookies here
                }
            );

            console.log("📦 Session Data Fetched:", sessionRes.data);

            setResult(sessionRes.data);
            navigate("/results");
        } catch (error) {
            console.error("❌ Submission Error:", error.response || error.message);
        }
    };

    return (
        <form onSubmit={handleSubmit} encType="multipart/form-data">
            <input name="name" placeholder="Name" onChange={handleChange} required />
            <input name="email" type="email" placeholder="Email" onChange={handleChange} required />
            <input name="contact" placeholder="Contact" onChange={handleChange} required />
            <input name="github" placeholder="GitHub URL" onChange={handleChange} />
            <input name="linkedin" placeholder="LinkedIn URL" onChange={handleChange} />
            <input type="file" name="resume" accept="application/pdf" onChange={handleFileChange} />
            <button type="submit">Submit</button>
        </form>
    );
};

export default FormComponent;
