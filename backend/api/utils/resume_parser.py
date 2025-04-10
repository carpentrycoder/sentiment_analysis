import re

keywords = [
    'Python', 'Java', 'C', 'JavaScript', 'Django', 'Flask', 'React', 'Machine Learning',
    'LLM', 'SQL', 'Pandas', 'Numpy', 'Seaborn', 'Bootstrap', 'GitHub', 'Tailwind',
    'C++', 'TypeScript', 'HTML', 'CSS', 'Tailwind CSS', 'React.js', 'Vue.js', 'Angular',
    'Node.js', 'Express.js', 'NumPy', 'Matplotlib', 'Plotly', 'Scikit-Learn', 'Deep Learning',
    'LLMs', 'Hugging Face', 'Transformers', 'TensorFlow', 'PyTorch', 'OpenCV', 'Power BI',
    'Excel', 'Git', 'Docker', 'PostgreSQL', 'MongoDB'
]

def basic_resume_parser(text):
    def extract_email(text):
        match = re.search(r'[\w\.-]+@[\w\.-]+', text)
        return match.group(0) if match else None

    def extract_phone(text):
        match = re.search(r'\+91[-\s]?[0-9]{10}|\b[7-9]\d{9}\b', text)
        return match.group(0) if match else None

    def extract_name(text):
        lines = text.split("\n")
        for line in lines:
            if "SUMMARY" in line.upper():
                return lines[lines.index(line) - 1].strip()
        return lines[0]  # fallback: return first line

    def extract_education(text):
        edu_keywords = ['university', 'college', 'bachelor', 'master', 'b.tech', 'm.tech', 'cgpa', 'degree']
        lines = text.lower().split("\n")
        return [line for line in lines if any(k in line for k in edu_keywords)]

    def extract_cgpa(edu_lines):
        for line in edu_lines:
            match = re.search(r'cgpa.*?(\d+\.\d+)', line)
            if match:
                return float(match.group(1))
        return None

    skills = [kw for kw in keywords if kw.lower() in text.lower()]
    education = extract_education(text)

    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "education": education,
        "cgpa": extract_cgpa(education),
        "skills": skills,
        "skills_count": len(skills)
    }
