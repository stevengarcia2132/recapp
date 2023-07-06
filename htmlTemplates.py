css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center; /* Added to center align text */
}

.chat-message.user {
    background-color: #2b313e;
}

.chat-message.bot {
    background-color: #475063;
}

.chat-message .avatar {
  width: 20%;
}

.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}

.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}

.chat-message .name {
  margin-top: 0.5rem;
  font-size: 0.8rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://companieslogo.com/img/orig/NSIT-65c6ad49.png?t=1600938003">
        <div class="name">Recapp Bot</div> <!-- Moved name element inside avatar div -->
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/1077/1077114.png">
        <div class="name">User</div> <!-- Moved name element inside avatar div -->
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
