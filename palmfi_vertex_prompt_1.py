# Import the required libraries and modules
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair

# Function to handle the chat interaction
def handle_chat_interaction(query):
    try:
        # Initialize the vertexai with project and location details
        vertexai.init(project="palm-fin-data", location="us-central1")

        # Load the pre-trained chat model
        chat_model = ChatModel.from_pretrained("chat-bison")

        # Set the parameters for generating responses
        parameters = {
            "candidate_count": 1,
            "max_output_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40
        }

        # Context
        context = (
            "The AI-native finance & connectivity platform is designed to assist users with insights, "
            "recommendations, and analytics derived from both structured and unstructured data. The platform "
            "integrates with bank accounts, commerce rails, and accounting/ERP systems to provide a holistic view "
            "of a user's financial health and business operations. Users can inquire about their financial status, "
            "transactions, vendors, and receive actionable insights to optimize their cash flow and expenses. The AI "
            "is also capable of understanding and responding to queries related to anomalies in financial data and offers "
            "personalized underwriting & origination solutions."
        )

        # Example interactions
        examples = [
            InputOutputTextPair(
                input_text="What are my recurring expenses and how can I save money?",
                output_text=(
                    "Your recurring expenses include rent ($2,000/month), utilities ($500/month), subscription services "
                    "($300/month), and employee salaries ($10,000/month). To save money, you could consider renegotiating "
                    "your lease terms, optimizing utility usage, evaluating the necessity of subscription services, and "
                    "exploring options for operational efficiency to potentially reduce labor costs."
                )
            ),
            InputOutputTextPair(
                input_text="Who were my top 5 vendors last month?",
                output_text=(
                    "Your top 5 vendors last month based on the total purchase amount were:\n"
                    "1. ABC Supplies - $3,000\n"
                    "2. XYZ Electronics - $2,500\n"
                    "3. DEF Manufacturing - $2,000\n"
                    "4. GHI Services - $1,500\n"
                    "5. JKL Wholesale - $1,000"
                )
            ),
            InputOutputTextPair(
                input_text="What was my cash balance last week?",
                output_text="Last week, your cash balance was $12,500. This is calculated from your total income of $15,000 and total expenses of $2,500 during that week."
            ),
            InputOutputTextPair(
                input_text="Why did I have lower cash last month?",
                output_text=(
                    "Last month, your cash was lower due to an increase in expenses. You had a total expense of $20,000, "
                    "which included a one-time equipment purchase of $8,000, and an unexpected maintenance cost of $2,000. "
                    "Your income remained steady at $15,000, leading to a lower net cash flow."
                )
            ),
            InputOutputTextPair(
                input_text="Is there any unusual activity in my account?",
                output_text=(
                    "I detected an unusual transaction pattern last week, where there were multiple high-value transactions "
                    "totaling $10,000 in a single day, which is not typical for your account. I recommend reviewing these "
                    "transactions to ensure their authenticity."
                )
            ),
            InputOutputTextPair(
                input_text="Schedule a meeting with my top vendors to discuss discounts",
                output_text=(
                    "I've scheduled a virtual meeting with ABC Supplies, XYZ Electronics, DEF Manufacturing, GHI Services, "
                    "and JKL Wholesale for next Thursday at 2 PM. You will receive a calendar invite and reminders ahead of the meeting."
                )
            )
        ]

        # Start a chat session with context and example interactions
        chat = chat_model.start_chat(context=context, examples=examples)

        # Send a message to the model and get the response
        response = chat.send_message(query, **parameters)

        # Print the model's response
        return response.text

    except vertexai.errors.RequestError as e:
        # Handle specific errors from the Vertex AI SDK
        print(f"An error occurred while processing the request: {e}")
        return "I'm sorry, but I'm unable to process your request at the moment. Please try again later."

    except Exception as e:
        # Handle other general errors
        print(f"An unexpected error occurred: {e}")
        return "I'm sorry, but an unexpected error occurred. Please try again later."

# Test the function with a user query
user_query = "who was my highest paid vendor & tell me any relevant info"
response = handle_chat_interaction(user_query)
print(f"Response from Model: {response}")
