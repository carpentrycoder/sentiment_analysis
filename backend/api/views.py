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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

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
            threshold = request.query_params.get("threshold", 10)
            try:
                threshold = float(threshold)
            except ValueError:
                threshold = 10

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
            if not resume_data or "skills" not in resume_data or "skills_count" not in resume_data or "cgpa" not in resume_data:
                return Response({"error": "Incomplete resume data."}, status=400)

            detected_skills = resume_data["skills"]
            skills_count = len(detected_skills)
            cgpa = resume_data["cgpa"]
            threshold = float(request.query_params.get("threshold", 50))

            # === PART 1: Career Suggestions ===
            csv_path = os.path.join(os.path.dirname(__file__), "generated_roles.csv")
            if not os.path.exists(csv_path):
                return Response({"error": "Role data file not found."}, status=500)
            
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

            # Simulated training data for ML model
            training_data = [{"matched_skills": i, "skills_count": 30, "match_percent": i / 30, "cgpa": round(6.0 + (i % 4) * 0.5, 2), "shortlisted": 1 if i > 20 else 0} for i in range(10, 35)]

            df = pd.DataFrame(training_data)
            X = df[["skills_count", "cgpa", "matched_skills", "match_percent"]]
            y = df["shortlisted"]

            model = LogisticRegression()
            model.fit(X, y)

            input_df = pd.DataFrame([{
                "skills_count": skills_count,
                "cgpa": cgpa,
                "matched_skills": matched_skills,
                "match_percent": match_ratio
            }])

            shortlisting_prob = round(model.predict_proba(input_df)[0][1], 4)
            confidence = "High" if shortlisting_prob >= 0.75 else "Moderate" if shortlisting_prob >= 0.5 else "Low"
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

class MarketValueView(APIView):
    def post(self, request):
        try:
            resume_data = request.data.get("resume_data")
            job_skills = request.data.get("job_skills")

            if not resume_data or not job_skills:
                return Response({"error": "Missing resume_data or job_skills"}, status=400)

            # --- Step 1: Feature Calculation ---
            matched_skills = len(set(resume_data["skills"]).intersection(set(job_skills)))
            match_ratio = matched_skills / len(job_skills)
            cgpa = resume_data.get("cgpa", 0.0)

            # Prepare input for fit score + label
            features_df = pd.DataFrame([{
                "matched_skills": matched_skills,
                "total_skills": resume_data["skills_count"],
                "match_ratio": match_ratio,
                "cgpa": cgpa
            }])

            # Train temporary classifiers (replace with saved models later)
            clf_data = []
            for i in range(10, 31):
                clf_data.append({
                    "matched_skills": i,
                    "total_skills": 30,
                    "match_ratio": i / 30,
                    "cgpa": round(6.0 + (i % 4) * 0.5, 2),
                    "fit_label": 1 if i > 20 else 0,
                    "fit_score": (i / 30) * 10
                })
            df_clf = pd.DataFrame(clf_data)
            clf = LogisticRegression()
            reg = LinearRegression()

            clf.fit(df_clf[["matched_skills", "total_skills", "match_ratio", "cgpa"]], df_clf["fit_label"])
            reg.fit(df_clf[["matched_skills", "total_skills", "match_ratio", "cgpa"]], df_clf["fit_score"])

            fit_label = int(clf.predict(features_df)[0])
            fit_score = float(round(reg.predict(features_df)[0], 2))

            # --- Step 2: Full Market Value Prediction Input ---
            resume_features = {
                "shortlisting_probability": resume_data.get("shortlisting_probability", 0.7),
                "matched_skills_count": matched_skills,
                "total_job_skills": len(job_skills),
                "skill_match_percent": match_ratio,
                "cgpa": cgpa,
                "recommendation": 1 if fit_score >= 5 else 0,
                "fit_label": fit_label,
                "fit_score": fit_score,
                "resume_quality_score": resume_data.get("resume_quality_score", 7.0)
            }

            # --- Step 3: Load & Train Market Value Model ---
            csv_path =  ("data/resume_market_value_dataset.csv")
            training_data = pd.read_csv(csv_path)

            required_cols = [
                "shortlisting_probability", "matched_skills_count", "total_job_skills",
                "skill_match_percent", "cgpa", "recommendation", "fit_label",
                "fit_score", "resume_quality_score", "market_value_score"
            ]
            missing = [col for col in required_cols if col not in training_data.columns]
            if missing:
                return Response({"error": f"Missing columns in CSV: {missing}"}, status=500)

            X_train = training_data.drop("market_value_score", axis=1)
            y_train = training_data["market_value_score"]

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # --- Step 4: Prediction & Insights ---
            input_df = pd.DataFrame([resume_features])
            predicted_score = int(model.predict(input_df)[0])

            perm = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
            importances = perm.importances_mean
            top_features = [X_train.columns[i] for i in np.argsort(importances)[::-1][:3]]

            if predicted_score >= 85:
                label = "🌟 Excellent Candidate"
                note = "Top-tier resume. Strong recommendation for hiring loop."
                hr_summary = "Candidate exhibits high potential, well-aligned with industry benchmarks."
                candidate_advice = "Maintain this quality and focus on upskilling in leadership or niche tools."
            elif predicted_score >= 70:
                label = "✅ Good Fit"
                note = "Ready for interview. Consider it for the next round."
                hr_summary = "Skills and resume strength indicate good job readiness."
                candidate_advice = "Continue improving project portfolio and soft skills for top-tier roles."
            elif predicted_score >= 50:
                label = "⚠️ Average Fit"
                note = "Potential with some gaps. Evaluate project depth or training needs."
                hr_summary = "Candidate meets minimum expectations but may require onboarding/training."
                candidate_advice = "Focus on strengthening projects, certifications, or tool mastery."
            else:
                label = "❌ Not Recommended"
                note = "Low readiness score. Skip unless the role is junior/entry-level."
                hr_summary = "Skillset or experience does not align with role requirements."
                candidate_advice = "Consider revising your resume, taking up internships, or certifications."

            breakdown = {
                "Shortlisting Probability": resume_features["shortlisting_probability"],
                "Skill Match %": resume_features["skill_match_percent"],
                "Matched Skills Count": resume_features["matched_skills_count"],
                "CGPA": resume_features["cgpa"],
                "Fit Score": resume_features["fit_score"],
                "Resume Quality Score": resume_features["resume_quality_score"]
            }

            return Response({
                "market_value_score": predicted_score,
                "label": label,
                "note": note,
                "top_factors": top_features,
                "score_breakdown": breakdown,
                "hr_summary": hr_summary,
                "candidate_advice": candidate_advice,
                "fit_label": fit_label,
                "fit_score": fit_score
            })

        except Exception as e:
            return Response({"error": str(e)}, status=500)