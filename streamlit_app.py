import streamlit as st
from pymongo import MongoClient
import time

# MongoDB Atlas connection
MONGODB_URI = "mongodb+srv://infernapeamber:g9kASflhhSQ26GMF@cluster0.mjoloub.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"  # Replace with your actual MongoDB Atlas URI
client = MongoClient(MONGODB_URI)
db = client['form_db']  # Database name
submissions_collection = db['submissions']  # Collection for form submissions

def save_submission(name: str, email: str, message: str) -> bool:
    """
    Save form submission to MongoDB Atlas.
    
    Args:
        name (str): User's name
        email (str): User's email
        message (str): User's message
    
    Returns:
        bool: True if submission was successful, False otherwise
    """
    try:
        submission = {
            "name": name,
            "email": email,
            "message": message,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        submissions_collection.insert_one(submission)
        return True
    except Exception as e:
        st.error(f"Failed to save submission: {str(e)}")
        return False

def get_all_submissions() -> list:
    """
    Retrieve all submissions from MongoDB Atlas.
    
    Returns:
        list: List of submission dictionaries
    """
    try:
        submissions = list(submissions_collection.find())
        # Remove MongoDB's internal _id field
        for submission in submissions:
            submission.pop('_id', None)
        return submissions
    except Exception as e:
        st.error(f"Failed to load submissions: {str(e)}")
        return []

# Streamlit app
st.title("📝 Simple Form with MongoDB Atlas")

st.write("Please fill out the form below. On submission, all stored data will be displayed.")

# Form
with st.form("submission_form"):
    name = st.text_input("Name", placeholder="Enter your name")
    email = st.text_input("Email", placeholder="Enter your email")
    message = st.text_area("Message", placeholder="Enter your message")
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        if name and email and message:
            if save_submission(name, email, message):
                st.success("Submission saved successfully!")
                
                # Display all submissions
                st.subheader("📋 All Submissions")
                submissions = get_all_submissions()
                
                if submissions:
                    for submission in submissions:
                        st.write("---")
                        st.write(f"**Name**: {submission['name']}")
                        st.write(f"**Email**: {submission['email']}")
                        st.write(f"**Message**: {submission['message']}")
                        st.write(f"**Submitted At**: {submission['submitted_at']}")
                else:
                    st.info("No submissions found in the database.")
        else:
            st.warning("Please fill in all fields!")

# Close MongoDB connection (optional, PyMongo handles connection pooling)
client.close()
