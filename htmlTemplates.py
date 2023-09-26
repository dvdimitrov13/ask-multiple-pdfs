css = '''
<style>
.chat-box {
    border: 1px solid #ccc;
    background-color: #ebebeb;  /* Mid-tone, adjust for light/dark mode */
    padding: 1rem;
    height: 400px;  /* Adjust based on your preference */
    overflow-y: auto;
}

.chat-message {
    padding: 0.8rem; 
    border-radius: 0.5rem; 
    margin: 0.5rem 0; 
    display: flex;
    align-items: center;
    max-width: 80%;
}

.chat-message.user {
    justify-content: flex-end;
    background-color: #dcf8c6;  /* Light green for user messages */
    margin-left: auto;
}

.chat-message.bot {
    justify-content: flex-start;
    background-color: #ebebeb;  /* Light gray for bot messages */
    margin-right: auto;
}

.chat-message .emoji {
    font-size: 1.5rem;
    margin: 0 0.5rem;
}

.chat-message .message {
    padding: 0.5rem 1rem;
    color: #333;
    border-radius: 0.5rem;
    background-color: #fff;
    word-wrap: break-word;
}

input[type="text"] {
    margin-top: 1rem;
    width: 100%;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="emoji">🤖</div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="emoji">🧑</div>    
    <div class="message">{{MSG}}</div>
</div>
'''




