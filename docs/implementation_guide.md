# 🛠️ AI Agent Implementation Guide

This guide provides step-by-step instructions for implementing the AI agent research participation system in NeuroPi.

## 🏗️ Architecture Overview

The AI agent system integrates with the existing NeuroPi Django application through:
- **New Models**: Research participation requests, study matching, notifications
- **Extended Views**: AI agent endpoints, conversation handling, response generation
- **Frontend Integration**: Chat interface, interactive buttons, real-time updates
- **Background Tasks**: Matching algorithms, notification processing, analytics

## 📋 Implementation Steps

### Step 1: Database Models Extension

Create new models in `user_management/models.py`:

```python
class ResearchParticipationRequest(models.Model):
    """Model for tracking research participation requests"""
    REQUEST_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('declined', 'Declined'),
        ('expired', 'Expired'),
        ('withdrawn', 'Withdrawn'),
    ]
    
    participant = models.ForeignKey(User, on_delete=models.CASCADE, related_name='participation_requests')
    researcher = models.ForeignKey(User, on_delete=models.CASCADE, related_name='received_requests', null=True, blank=True)
    request_type = models.CharField(max_length=20, choices=[('open', 'Open Invitation'), ('specific', 'Specific Researcher')])
    status = models.CharField(max_length=20, choices=REQUEST_STATUS_CHOICES, default='pending')
    message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField()

class StudyInvitation(models.Model):
    """Model for study invitations from researchers to participants"""
    researcher = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sent_invitations')
    participant = models.ForeignKey(User, on_delete=models.CASCADE, related_name='received_invitations')
    study_title = models.CharField(max_length=200)
    study_description = models.TextField()
    compensation = models.DecimalField(max_digits=10, decimal_places=2)
    session_count = models.IntegerField()
    total_duration = models.DurationField()
    status = models.CharField(max_length=20, choices=[('sent', 'Sent'), ('accepted', 'Accepted'), ('declined', 'Declined')])
    personal_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    response_deadline = models.DateTimeField()

class AIAgentConversation(models.Model):
    """Model for tracking AI agent conversations"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ai_conversations')
    session_id = models.UUIDField(default=uuid.uuid4)
    conversation_type = models.CharField(max_length=50)
    current_step = models.CharField(max_length=100)
    context_data = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

### Step 2: AI Agent Views

Create `user_management/ai_agent_views.py`:

```python
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import json
from .ai_agent_engine import AIAgentEngine

@login_required
@csrf_exempt
def ai_agent_chat(request):
    """Main endpoint for AI agent conversations"""
    if request.method == 'POST':
        data = json.loads(request.body)
        message = data.get('message', '')
        session_id = data.get('session_id', '')
        
        # Initialize AI agent engine
        agent = AIAgentEngine(user=request.user, session_id=session_id)
        
        # Process user message and generate response
        response = agent.process_message(message)
        
        return JsonResponse(response)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@login_required
def research_participation_entry(request):
    """Entry point for research participation flow"""
    # Check user role and eligibility
    if not is_participant(request.user):
        return redirect('home')
    
    # Initialize AI agent conversation
    agent = AIAgentEngine(user=request.user)
    initial_response = agent.start_participation_flow()
    
    context = {
        'initial_response': initial_response,
        'user_profile': request.user.participant_profile,
        'consent_status': check_consent_status(request.user),
    }
    
    return render(request, 'user_management/ai_agent_chat.html', context)
```

### Step 3: AI Agent Engine

Create `user_management/ai_agent_engine.py`:

```python
class AIAgentEngine:
    """Core AI agent processing engine"""
    
    def __init__(self, user, session_id=None):
        self.user = user
        self.session_id = session_id or str(uuid.uuid4())
        self.conversation = self.get_or_create_conversation()
    
    def process_message(self, message):
        """Process user message and generate appropriate response"""
        # Determine current flow and step
        current_flow = self.conversation.conversation_type
        current_step = self.conversation.current_step
        
        # Route to appropriate handler
        if current_flow == 'participation_request':
            return self.handle_participation_flow(message, current_step)
        elif current_flow == 'study_browsing':
            return self.handle_study_browsing(message, current_step)
        elif current_flow == 'ai_matching':
            return self.handle_ai_matching(message, current_step)
        else:
            return self.handle_general_query(message)
    
    def start_participation_flow(self):
        """Initialize research participation conversation"""
        # Check user eligibility
        eligibility = self.check_user_eligibility()
        
        if not eligibility['eligible']:
            return self.generate_eligibility_response(eligibility)
        
        # Generate welcome message and options
        return self.generate_participation_options()
    
    def generate_response(self, message_type, content, actions=None, context=None):
        """Generate structured AI agent response"""
        return {
            'agent_response': {
                'message_type': message_type,
                'content': content,
                'actions': actions or [],
                'context': context or {},
                'personalization': {
                    'user_name': self.user.first_name or self.user.username,
                    'participation_level': self.get_participation_level(),
                }
            }
        }
```

### Step 4: Frontend Integration

Create `templates/user_management/ai_agent_chat.html`:

```html
{% extends 'layout.html' %}
{% load static %}

{% block title %}Research Participation Assistant{% endblock %}

{% block content %}
<div class="container-fluid">
    <div class="row">
        <div class="col-md-8 mx-auto">
            <div class="ai-chat-container">
                <div class="chat-header">
                    <h3><i class="bi bi-robot"></i> NeuroPi Research Assistant</h3>
                    <p class="text-muted">I'm here to help you participate in research studies</p>
                </div>
                
                <div id="chat-messages" class="chat-messages">
                    <!-- Initial AI response -->
                    <div class="message ai-message">
                        <div class="message-content">
                            {{ initial_response.content|safe }}
                        </div>
                        <div class="message-actions">
                            {% for action in initial_response.actions %}
                            <button class="btn btn-outline-primary action-btn" 
                                    data-action="{{ action.action_type }}" 
                                    data-target="{{ action.target }}">
                                {{ action.label }}
                            </button>
                            {% endfor %}
                        </div>
                    </div>
                </div>
                
                <div class="chat-input">
                    <div class="input-group">
                        <input type="text" id="user-input" class="form-control" 
                               placeholder="Type your message or click a button above...">
                        <button class="btn btn-primary" id="send-btn">
                            <i class="bi bi-send"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
// AI Agent Chat JavaScript
class AIAgentChat {
    constructor() {
        this.sessionId = '{{ session_id }}';
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        document.getElementById('send-btn').addEventListener('click', () => this.sendMessage());
        document.getElementById('user-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        
        // Action button handlers
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('action-btn')) {
                this.handleActionButton(e.target);
            }
        });
    }
    
    async sendMessage() {
        const input = document.getElementById('user-input');
        const message = input.value.trim();
        if (!message) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        input.value = '';
        
        // Send to AI agent
        const response = await this.callAIAgent(message);
        this.handleAIResponse(response);
    }
    
    async callAIAgent(message) {
        const response = await fetch('/user_management/ai-agent/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({
                message: message,
                session_id: this.sessionId
            })
        });
        return await response.json();
    }
    
    handleAIResponse(response) {
        const agentResponse = response.agent_response;
        this.addMessage(agentResponse.content, 'ai', agentResponse.actions);
    }
    
    addMessage(content, sender, actions = []) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        messageDiv.innerHTML = `
            <div class="message-content">${content}</div>
            ${actions.length > 0 ? `
                <div class="message-actions">
                    ${actions.map(action => `
                        <button class="btn btn-outline-primary action-btn" 
                                data-action="${action.action_type}" 
                                data-target="${action.target}">
                            ${action.label}
                        </button>
                    `).join('')}
                </div>
            ` : ''}
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// Initialize chat when page loads
document.addEventListener('DOMContentLoaded', () => {
    new AIAgentChat();
});
</script>
{% endblock %}
```

### Step 5: URL Configuration

Add to `user_management/urls.py`:

```python
urlpatterns = [
    # ... existing URLs ...
    path('research/participate/', views.research_participation_entry, name='research_participation'),
    path('ai-agent/chat/', ai_agent_views.ai_agent_chat, name='ai_agent_chat'),
    path('studies/browse/', views.browse_studies, name='browse_studies'),
    path('studies/apply/<int:study_id>/', views.apply_to_study, name='apply_to_study'),
]
```

### Step 6: Navigation Integration

Update `templates/layout.html` to add research participation link:

```html
<!-- Add to participant dropdown menu -->
{% if user.profile.role == 'user' %}
<li><a class="dropdown-item" href="{% url 'research_participation' %}">
    <i class="bi bi-search"></i> Join Research Study
</a></li>
{% endif %}
```

## 🎨 Styling and UX

Add CSS for chat interface in `static/css/ai_agent.css`:

```css
.ai-chat-container {
    max-width: 800px;
    margin: 0 auto;
    background: white;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    overflow: hidden;
}

.chat-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    text-align: center;
}

.chat-messages {
    height: 500px;
    overflow-y: auto;
    padding: 20px;
    background: #f8f9fa;
}

.message {
    margin-bottom: 20px;
    animation: fadeIn 0.3s ease-in;
}

.ai-message .message-content {
    background: white;
    padding: 15px;
    border-radius: 15px 15px 15px 5px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.user-message .message-content {
    background: #007bff;
    color: white;
    padding: 15px;
    border-radius: 15px 15px 5px 15px;
    margin-left: auto;
    max-width: 70%;
}

.message-actions {
    margin-top: 10px;
}

.action-btn {
    margin: 5px;
    border-radius: 20px;
    transition: all 0.3s ease;
}

.action-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
```

## 🧪 Testing Strategy

### Unit Tests
Create `user_management/tests/test_ai_agent.py`:

```python
from django.test import TestCase, Client
from django.contrib.auth.models import User
from accounts.models import UserProfile
from user_management.ai_agent_engine import AIAgentEngine

class AIAgentEngineTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('testuser', 'test@example.com', 'password')
        UserProfile.objects.create(user=self.user, role='user')
        self.agent = AIAgentEngine(user=self.user)
    
    def test_participation_flow_initialization(self):
        response = self.agent.start_participation_flow()
        self.assertIn('agent_response', response)
        self.assertIn('content', response['agent_response'])
    
    def test_eligibility_check(self):
        eligibility = self.agent.check_user_eligibility()
        self.assertIn('eligible', eligibility)
        self.assertIsInstance(eligibility['eligible'], bool)
```

### Integration Tests
Test the complete flow from frontend to backend:

```python
class AIAgentIntegrationTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user('testuser', 'test@example.com', 'password')
        UserProfile.objects.create(user=self.user, role='user')
        self.client.login(username='testuser', password='password')
    
    def test_research_participation_entry(self):
        response = self.client.get('/user_management/research/participate/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Research Assistant')
    
    def test_ai_agent_chat_endpoint(self):
        response = self.client.post('/user_management/ai-agent/chat/', 
                                  json.dumps({'message': 'I want to join a study'}),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('agent_response', data)
```

## 🚀 Deployment Checklist

- [ ] Run database migrations
- [ ] Update static files
- [ ] Configure environment variables
- [ ] Set up background task processing
- [ ] Enable logging and monitoring
- [ ] Test all user flows
- [ ] Verify security measures
- [ ] Update documentation

## 📊 Monitoring & Analytics

Implement tracking for:
- Conversation completion rates
- User satisfaction scores
- Feature usage statistics
- Error rates and types
- Performance metrics

This implementation guide provides the foundation for building the AI agent system. Customize and extend based on specific requirements and user feedback.
