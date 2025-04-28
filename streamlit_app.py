import streamlit as st
from pymongo import MongoClient
import time
import urllib.parse

# MongoDB Atlas connection
# Replace with your actual MongoDB Atlas URI
# Use either mongodb+srv or standard URI format
MONGODB_URI = "mongodb+srv://infernapeamber:g9kASflhhSQ26GMF@cluster0.mjoloub.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Optional: Standard URI fallback if mongodb+srv fails
# Example: mongodb://<username>:<password>@ac-4hozb1p-shard-00-00.mjoloub.mongodb.net:27017,...
# MONGODB_URI = "your_standard_mongodb_uri_here"

# Initialize MongoDB client
try:
    # URL-encode username and password for safety
    parsed_uri = urllib.parse.urlparse(MONGODB_URI)
    if parsed_uri.scheme == "mongodb+srv":
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000
        )
    else:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            ssl=True
        )
    
    # Test connection
    client.admin.command('ping')
    db = client['form_db']  # Database name
    submissions_collection = db['submissions']  # Collection for form submissions
    st.success("Connected to MongoDB Atlas successfully!")
except Exception as e:
    st.error(f"Failed to connect to MongoDB Atlas: {str(e)}")
    st.write("**Troubleshooting Steps**:")
    st.write("1. **Verify MongoDB Atlas URI**: Ensure username, password, and cluster name are correct. URL-encode special characters in the password (e.g., @ → %40).")
    st.write("2. **Network Access**: In MongoDB Atlas, set Network Access to 0.0.0.0/0 (allow all) for testing, especially for Streamlit Cloud.")
    st.write("3. **TLS Support**: Ensure Python 3.9+ and pymongo 4.8.0+ are used. Check Streamlit Cloud settings.")
    st.write("4. **Try Standard URI**: If mongodb+srv fails, use the standard URI from Atlas Connect > Drivers.")
    st.write("5. **Cluster Status**: Verify the cluster is running in MongoDB Atlas (not paused).")
    st.write("6. **Test Locally**: Run the app locally to isolate cloud-specific issues.")
    st.stop()

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

st.write("Fill out the form below to submit data to MongoDB Atlas. All submissions will be displayed upon successful submission.")

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
