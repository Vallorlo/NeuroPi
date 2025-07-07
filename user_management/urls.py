from django.urls import path
from . import views

app_name = 'user_management'

urlpatterns = [
    # Dashboard
    path('', views.dashboard, name='dashboard'),
    
    # Participant management
    path('participants/', views.participant_list, name='participant_list'),
    path('participants/<uuid:participant_id>/', views.participant_detail, name='participant_detail'),
    
    # Consent management
    path('consent/', views.consent_management, name='consent_management'),
    path('consent/create/', views.create_consent_form, name='create_consent_form'),
    path('consent/<int:form_id>/edit/', views.edit_consent_form, name='edit_consent_form'),
    path('consent/<int:form_id>/view/', views.view_consent_form, name='view_consent_form'),
    path('consent/<int:form_id>/assign/', views.assign_consent_form, name='assign_consent_form'),
    path('consent/assign/', views.assign_consent_form, name='assign_consent_form_general'),

    # Consent signing and downloads
    path('consent/review/<int:assignment_id>/', views.review_consent_assignment, name='review_consent_assignment'),
    path('consent/sign/<int:record_id>/', views.sign_consent, name='sign_consent'),
    path('consent/withdraw/<int:record_id>/', views.withdraw_consent, name='withdraw_consent'),
    path('consent/download/<int:record_id>/', views.download_consent_pdf, name='download_consent_pdf'),
    
    # Profile management
    path('profile/', views.edit_profile, name='edit_profile'),
    path('profile/researcher/', views.edit_researcher_profile, name='edit_researcher_profile'),
    path('workflow-guide/', views.workflow_guide, name='workflow_guide'),
    
    # Research participation
    path('join-research-study/', views.join_research_study, name='join_research_study'),
    path('submit-participation-request/', views.submit_participation_request, name='submit_participation_request'),
    path('participation/request/', views.create_participation_request, name='create_participation_request'),
    path('participation/requests/', views.participation_requests, name='participation_requests'),
    path('participation/researcher-requests/', views.researcher_requests, name='researcher_requests'),
    path('participation/respond/<int:request_id>/', views.respond_to_request, name='respond_to_request'),
    path('confirm-participation/<int:request_id>/', views.confirm_participation, name='confirm_participation'),

    # Eligible participants and study invitations
    path('eligible-participants/', views.eligible_participants_view, name='eligible_participants_view'),
    path('invite-participant/<uuid:participant_id>/', views.invite_participant_to_study, name='invite_participant_to_study'),

    # Availability management
    path('availability/', views.manage_availability, name='manage_availability'),

    # Session management
    path('sessions/', views.participant_sessions, name='participant_sessions'),
    path('sessions/schedule/', views.schedule_session, name='schedule_session'),



    # API endpoints
    path('api/participants/', views.participant_list_api, name='participant_list_api'),
    path('api/consent-status/', views.consent_status_api, name='consent_status_api'),
]
