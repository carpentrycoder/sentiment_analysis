from rest_framework import generics
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .models import InputData
from .serializers import InputDataSerializer
from .utils.resume_parser import basic_resume_parser
import PyPDF2
import os 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class InputDataListCreate(generics.ListCreateAPIView):
    queryset = InputData.objects.all()
    serializer_class = InputDataSerializer
    parser_classes = (MultiPartParser, FormParser)  # Handle file uploads

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)

        if serializer.is_valid():
            instance = serializer.save()

            # Store LinkedIn & GitHub URLs in Session
            request.session['linkedin'] = instance.linkedin if instance.linkedin else None
            request.session['github'] = instance.github if instance.github else None

            # Extract Resume Text & Store in Database
            if instance.resume:
                text = self.extract_text_from_pdf(instance.resume)
                instance.resume_text = text  # ✅ Save extracted text in DB
                instance.save()

                request.session['resume_text'] = text  # ✅ Store in session too
                print("📄 Extracted Resume Text:", text)

                # ✅ Extract structured data using basic parser
                parsed_data = basic_resume_parser(text)
                request.session['resume_info'] = parsed_data
                print("🧠 Parsed Resume Info:", parsed_data)

            # ✅ Force session save
            request.session.modified = True
            request.session.save()

            # Debugging
            print("🟢 Stored in Session:")
            print("🔹 LinkedIn:", request.session.get('linkedin'))
            print("🔹 GitHub:", request.session.get('github'))
            print("🔹 Resume Text:", request.session.get('resume_text'))
            print("🔹 Resume Info:", request.session.get('resume_info'))

            return Response(serializer.data, status=status.HTTP_201_CREATED)

        print("❌ Validation Errors:", serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a PDF file"""
        text = ""
        try:
            print(f"📂 Trying to read PDF: {pdf_file.name}")
            with pdf_file.open('rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"📄 Total Pages in PDF: {num_pages}")
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    print(f"📝 Extracted Text from Page {i+1}: {page_text[:200]}")
            if not text:
                print("❌ No text extracted! The PDF may be an image or encrypted.")
        except Exception as e:
            print(f"⚠️ Error extracting text: {e}")
        return text


class SessionDataView(APIView):
    def get(self, request, *args, **kwargs):
        print("🔎 Checking Session Data in GET Request:")
        print("🔹 LinkedIn:", request.session.get('linkedin', 'Not available'))
        print("🔹 GitHub:", request.session.get('github', 'Not available'))
        print("🔹 Resume Text:", request.session.get('resume_text', 'No resume text stored'))
        print(f"Session Keys: {request.session.keys()}")
        print(f"Full Session Data: {dict(request.session.items())}")

        return Response({
            'linkedin': request.session.get('linkedin', 'Not available'),
            'github': request.session.get('github', 'Not available'),
            'resume_text': request.session.get('resume_text', 'No resume text stored'),
            'resume_info': request.session.get('resume_info', {})
        })


class ResumeParsedInfoView(APIView):
    def get(self, request, *args, **kwargs):
        resume_info = request.session.get('resume_info', {})
        return Response(resume_info)


class CareerSuggestionsView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            # Load resume skills from session
            resume_info = request.session.get('resume_info')
            if not resume_info or 'skills' not in resume_info:
                return Response({"error": "No parsed resume info or skills found in session."},
                                status=status.HTTP_400_BAD_REQUEST)

            detected_skills = resume_info["skills"]

            # Get threshold from query params
            threshold = request.query_params.get("threshold", 50)
            try:
                threshold = float(threshold)
            except ValueError:
                threshold = 50

            # Load roles CSV
            csv_path = os.path.join(os.path.dirname(__file__), 'generated_roles.csv')
            roles_df = pd.read_csv(csv_path)

            def match_score(user_skills, role_skills):
                return len(set(user_skills).intersection(set(role_skills))) / len(role_skills)

            suggestions = []
            for _, row in roles_df.iterrows():
                role = row["Role"]
                role_skills = [skill.strip() for skill in row["Skills"].split(",")]
                score = match_score(detected_skills, role_skills)
                match_percent = round(score * 100, 2)
                if match_percent >= threshold:
                    suggestions.append({
                        "career_role": role,
                        "match_percent": match_percent,
                        "required_skills": role_skills
                    })

            # Sort suggestions by match % descending
            suggestions = sorted(suggestions, key=lambda x: x["match_percent"], reverse=True)

            return Response({
                "threshold": threshold,
                "detected_skills": detected_skills,
                "suggestions": suggestions
            })

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

class ShortlistingPredictionView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            resume_data = request.session.get("resume_info")
            detected_skills = resume_data.get("skills") if resume_data else []

            if not resume_data or not detected_skills:
                return Response({"error": "Resume data or detected skills not found in session."}, status=400)

            # We'll use the same detected skills as the "job_skills" for matching
            job_skills = detected_skills

            matched_skills = len(set(resume_data["skills"]).intersection(set(job_skills)))
            match_ratio = matched_skills / len(job_skills)
            cgpa = resume_data.get("cgpa", 0.0)

            # Simulated training data for Logistic Regression
            data = []
            for i in range(10, 35):
                data.append({
                    "matched_skills": i,
                    "skills_count": 30,
                    "match_percent": i / 30,
                    "cgpa": round(6.0 + (i % 4) * 0.5, 2),
                    "shortlisted": 1 if i > 20 else 0
                })

            df = pd.DataFrame(data)
            X = df[["skills_count", "cgpa", "matched_skills", "match_percent"]]
            y = df["shortlisted"]

            # Train the logistic regression model
            model = LogisticRegression()
            model.fit(X, y)

            # Prediction input for the candidate
            prediction_input = pd.DataFrame([{
                "skills_count": resume_data["skills_count"],
                "cgpa": cgpa,
                "matched_skills": matched_skills,
                "match_percent": match_ratio
            }])

            shortlisting_prob = model.predict_proba(prediction_input)[0][1]
            shortlisting_prob = round(shortlisting_prob, 4)

            # Confidence & recommendation
            confidence = (
                "High" if shortlisting_prob >= 0.75 else
                "Moderate" if shortlisting_prob >= 0.5 else
                "Low"
            )
            recommendation = "Yes" if shortlisting_prob >= 0.5 else "No"

            return Response({
                "shortlisting_probability": shortlisting_prob,
                "matched_skills": matched_skills,
                "total_job_skills": len(job_skills),
                "match_percent": round(match_ratio * 100, 2),
                "cgpa": cgpa,
                "confidence": confidence,
                "recommended": recommendation,
                "used_skills": job_skills
            })

        except Exception as e:
            return Response({"error": str(e)}, status=500)


class ResumeInsightsView(APIView):
    def get(self, request):
        try:
            resume_data = request.session.get("resume_info")
            if not resume_data or "skills" not in resume_data:
                return Response({"error": "Resume data with skills not found in session."}, status=400)

            detected_skills = resume_data["skills"]
            threshold = float(request.query_params.get("threshold", 50))

            # === PART 1: Career Suggestions ===
            csv_path = os.path.join(os.path.dirname(__file__), "generated_roles.csv")
            roles_df = pd.read_csv(csv_path)

            def match_score(user_skills, role_skills):
                return len(set(user_skills).intersection(set(role_skills))) / len(role_skills)

            suggestions = []
            for _, row in roles_df.iterrows():
                role = row["Role"]
                role_skills = [skill.strip() for skill in row["Skills"].split(",")]
                score = match_score(detected_skills, role_skills)
                match_percent = round(score * 100, 2)
                if match_percent >= threshold:
                    suggestions.append({
                        "career_role": role,
                        "match_percent": match_percent,
                        "required_skills": role_skills
                    })

            suggestions = sorted(suggestions, key=lambda x: x["match_percent"], reverse=True)

            # === PART 2: Shortlisting Prediction ===
            job_skills = detected_skills
            matched_skills = len(set(detected_skills).intersection(set(job_skills)))
            match_ratio = matched_skills / len(job_skills)
            cgpa = resume_data.get("cgpa", 0.0)

            # Simulated training data for ML model
            training_data = []
            for i in range(10, 35):
                training_data.append({
                    "matched_skills": i,
                    "skills_count": 30,
                    "match_percent": i / 30,
                    "cgpa": round(6.0 + (i % 4) * 0.5, 2),
                    "shortlisted": 1 if i > 20 else 0
                })

            df = pd.DataFrame(training_data)
            X = df[["skills_count", "cgpa", "matched_skills", "match_percent"]]
            y = df["shortlisted"]

            model = LogisticRegression()
            model.fit(X, y)

            input_df = pd.DataFrame([{
                "skills_count": resume_data["skills_count"],
                "cgpa": cgpa,
                "matched_skills": matched_skills,
                "match_percent": match_ratio
            }])

            shortlisting_prob = round(model.predict_proba(input_df)[0][1], 4)
            confidence = (
                "High" if shortlisting_prob >= 0.75 else
                "Moderate" if shortlisting_prob >= 0.5 else
                "Low"
            )
            recommended = "Yes" if shortlisting_prob >= 0.5 else "No"

            return Response({
                "threshold": threshold,
                "detected_skills": detected_skills,
                "career_suggestions": suggestions,
                "shortlisting": {
                    "shortlisting_probability": shortlisting_prob,
                    "matched_skills": matched_skills,
                    "total_job_skills": len(job_skills),
                    "match_percent": round(match_ratio * 100, 2),
                    "cgpa": cgpa,
                    "confidence": confidence,
                    "recommended": recommended
                }
            })

        except Exception as e:
            return Response({"error": str(e)}, status=500)