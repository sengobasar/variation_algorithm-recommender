import streamlit as st
from streamlit_chat import message
from algorithm_recommender import AlgorithmRecommender

def main():
    st.set_page_config(
        page_title="🤖 ML Algorithm Recommender",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 ML Algorithm Recommender")
    st.write("Chat with me to get personalized ML algorithm recommendations!")
    
    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state.history = [
            {"role": "assistant", "content": "👋 Hi! I'm your ML Algorithm Assistant. Let's find the best algorithms for your project!"},
            {"role": "assistant", "content": "What type of machine learning problem are you working on? (binary classification/multiclass classification/regression/clustering)"}
        ]
    
    # Display chat messages
    for i, msg in enumerate(st.session_state.history):
        message(msg["content"], is_user=msg["role"] == "user", key=f"msg_{i}")
    
    # Chat input with icon
    with st.container():
        col1, col2 = st.columns([10, 1])
        with col1:
            user_input = st.text_input("Your response:", "", key="user_input", 
                                     placeholder="Type your response here...")
        with col2:
            st.write("")
            st.write("")
            if st.button("➡️"):
                if user_input.strip():
                    process_user_input(user_input)
    
    # Process Enter key
    if user_input and st.session_state.get("last_input") != user_input:
        st.session_state.last_input = user_input
        if user_input.strip():
            process_user_input(user_input)

def process_user_input(user_input: str):
    """Process user input and generate response."""
    # Add user message to history
    st.session_state.history.append({"role": "user", "content": user_input})
    
    # Get or initialize conversation state
    if 'conversation_state' not in st.session_state:
        st.session_state.conversation_state = {
            'step': 0,
            'params': {}
        }
    
    # Process conversation
    state = st.session_state.conversation_state
    response = ""
    
    if state['step'] == 0:  # Problem type
        state['params']['problem_type'] = user_input.lower()
        response = "What's the size of your dataset? (small < 10k samples, medium 10k-1M, large > 1M)"
        state['step'] = 1
    
    elif state['step'] == 1:  # Dataset size
        state['params']['dataset_size'] = user_input.lower()
        response = "What type of features does your data have? (numerical/categorical/mixed)"
        state['step'] = 2
    
    elif state['step'] == 2:  # Feature types
        state['params']['feature_types'] = user_input.lower()
        response = "Is your dataset balanced? (balanced/slightly imbalanced/highly imbalanced)"
        state['step'] = 3
    
    elif state['step'] == 3:  # Class balance
        state['params']['class_balance'] = user_input.lower()
        response = "What's your priority? (accuracy/interpretability/balanced accuracy & interpretability)"
        state['step'] = 4
    
    elif state['step'] == 4:  # Priority
        state['params']['priority'] = user_input.lower()
        # Get recommendations
        recommender = AlgorithmRecommender()
        params = {
            'problem_type': state['params'].get('problem_type', 'binary classification'),
            'dataset_size': state['params'].get('dataset_size', 'medium'),
            'feature_types': state['params'].get('feature_types', 'mixed'),
            'class_balance': state['params'].get('class_balance', 'balanced'),
            'priority': state['params'].get('priority', 'balanced accuracy & interpretability'),
            'metric': 'F1-score' if 'imbalanced' in state['params'].get('class_balance', '') else 'accuracy'
        }
        
        try:
            algorithms = recommender.recommend_algorithms(**params)
            response = f"Based on your inputs, I recommend these algorithms:\n"
            response += "\n".join([f"- {algo}" for algo in algorithms])
            response += "\n\nWould you like to start over? (yes/no)"
            state['step'] = 5
        except Exception as e:
            response = f"I encountered an error: {str(e)}. Let's start over."
            reset_conversation()
    
    elif state['step'] == 5:  # Restart conversation
        if user_input.lower().startswith('y'):
            reset_conversation()
            response = "Great! What type of machine learning problem are you working on? (binary classification/multiclass classification/regression/clustering)"
        else:
            response = "Thank you for using the ML Algorithm Recommender!"
    
    # Add assistant response to history
    st.session_state.history.append({"role": "assistant", "content": response})
    
    # Rerun to update the chat
    st.experimental_rerun()

def reset_conversation():
    """Reset the conversation state."""
    st.session_state.conversation_state = {
        'step': 0,
        'params': {}
    }
    st.session_state.history = [
        {"role": "assistant", "content": "👋 Hi! I'm your ML Algorithm Assistant. Let's find the best algorithms for your project!"},
        {"role": "assistant", "content": "What type of machine learning problem are you working on? (binary classification/multiclass classification/regression/clustering)"}
    ]

if __name__ == "__main__":
    main()
