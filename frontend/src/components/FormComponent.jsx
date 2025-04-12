import { useState } from "react";
import axios from "axios";
import "../app.css";
import { Link, useNavigate } from "react-router-dom"; // ✅ Correct import

const ProfileForm = ({ setResult }) => {
    const [formData, setFormData] = useState({
        name: "",
        email: "",
        contact: "",
        linkedin: "",
        github: "",
        resume: null
    });

    const [showAssistant, setShowAssistant] = useState(false);
    const navigate = useNavigate();

    const handleFileChange = (e) => {
        setFormData({ ...formData, resume: e.target.files[0] });
    };

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({ ...formData, [name]: value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!formData.name || !formData.email || !formData.contact) return;

        const formDataToSend = new FormData();
        Object.keys(formData).forEach((key) => {
            if (key === "resume" && formData[key]) {
                formDataToSend.append(key, formData[key], formData[key].name);
            } else {
                formDataToSend.append(key, formData[key]);
            }
        });

        try {
            await axios.post(
                "http://127.0.0.1:8000/api/data/",
                formDataToSend,
                {
                    headers: { "Content-Type": "multipart/form-data" },
                    withCredentials: true,
                }
            );

            // ✅ Navigate after successful response
            navigate("/results");

        } catch (error) {
            console.error("❌ Error submitting data:", error.response?.data || error.message);
            setResult({ error: "Failed to submit the form" });
        }
    };

    return (
        <>
            <div className={`main-container ${showAssistant ? "blurred" : ""}`}>
                <span>
                    <h2>Resume Analyzer</h2>
                </span>

                <form onSubmit={handleSubmit}>
                    <input
                        type="text"
                        name="name"
                        placeholder="Name"
                        value={formData.name}
                        onChange={handleChange}
                        required
                    />
                    <input
                        type="email"
                        name="email"
                        placeholder="Email"
                        value={formData.email}
                        onChange={handleChange}
                        required
                    />
                    <input
                        type="tel"
                        name="contact"
                        placeholder="Contact No."
                        value={formData.contact}
                        onChange={handleChange}
                        required
                    />
                    <input
                        type="url"
                        name="linkedin"
                        placeholder="LinkedIn URL"
                        value={formData.linkedin}
                        onChange={handleChange}
                    />
                    <input
                        type="url"
                        name="github"
                        placeholder="GitHub URL"
                        value={formData.github}
                        onChange={handleChange}
                    />
                    <input
                        type="file"
                        name="resume"
                        accept="application/pdf"
                        onChange={handleFileChange}
                    />
                    <button type="submit">Submit</button>
                </form>
            </div>

            {/* Floating AI Assistant Icon */}
            <img
                className="ai-robot-icon"
                src="https://cdn-icons-png.flaticon.com/128/10817/10817417.png"
                alt="AI Robot"
                onClick={() => setShowAssistant(!showAssistant)}
            />

            {/* AI Chat Assistant Window */}
            {showAssistant && (
                <div className="chat-window">
                    <div className="chat-header">
                        <img
                            src="https://cdn-icons-png.flaticon.com/128/10817/10817417.png"
                            alt="Chatbot"
                            className="chatbot-avatar"
                        />
                        <h3>AI Assistant</h3>
                        <span
                            className="close-button"
                            onClick={() => setShowAssistant(false)}
                        >
                            ✖
                        </span>
                    </div>
                    <div className="chat-body">
                        <div className="chat-message bot">
                            Hello! How can I assist you with your resume today?
                        </div>
                    </div>
                    <div className="chat-input">
                        <input type="text" placeholder="Type your message..." />
                        <button>Send</button>
                    </div>
                </div>
            )}
        </>
    );
};

export default ProfileForm;
