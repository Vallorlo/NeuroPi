from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.utils import timezone
from datetime import timedelta
from django.core.paginator import Paginator
from django.db.models import Q, Count, Max, F
from .models import (
    ParticipantProfile, ResearcherProfile, ConsentForm, ConsentRecord,
    ResearchParticipationRequest, ParticipantAvailability, ParticipantSession,
    ConsentAssignment, ConsentAuditLog
)
from .forms import (
    ParticipantProfileForm, ConsentFormForm, ConsentRecordForm, ResearcherProfileForm,
    DigitalSignatureForm, ConsentSigningForm, ResearchParticipationRequestForm, ParticipantAvailabilityForm,
    ResearcherResponseForm, ConsentAssignmentForm, ParticipantSessionForm
)
from .email_notifications import (
    notify_participation_request, notify_consent_assignment,
    notify_request_response, notify_session_reminder
)
from accounts.models import UserProfile
import json
from datetime import datetime, timedelta

@login_required
def dashboard(request):
    """Main dashboard for user management"""
    user_profile = getattr(request.user, 'profile', None)

    if not user_profile:
        messages.error(request, "User profile not found. Please contact an administrator.")
        return redirect('home')

    context = {
        'user_profile': user_profile,
    }

    if user_profile.is_trainer:
        # Researcher dashboard
        try:
            researcher_profile = request.user.researcher_profile

            # Get comprehensive consent assignment data
            all_assignments = ConsentAssignment.objects.filter(
                researcher=researcher_profile
            ).select_related('participant__user', 'consent_form').order_by('-assigned_at')

            recent_assignments = all_assignments[:10]
            pending_assignments = all_assignments.filter(status__in=['assigned', 'sent', 'viewed'])
            signed_assignments = all_assignments.filter(status='signed')

            # Get participants with active consent forms
            consented_participants = ParticipantProfile.objects.filter(
                consent_records__status='signed',
                consent_records__expires_at__gt=timezone.now(),
                is_active=True
            ).distinct().select_related('user')

            # Get recently signed consent forms (last 7 days)
            recent_signed_consents = ConsentRecord.objects.filter(
                status='signed',
                signed_at__gte=timezone.now() - timedelta(days=7)
            ).select_related('participant__user', 'consent_form').order_by('-signed_at')

            # Get newly signed consents from this researcher's assignments
            # Get all signed assignments from this researcher
            signed_assignments = ConsentAssignment.objects.filter(
                researcher=researcher_profile,
                status='signed'
            ).select_related('participant__user', 'consent_form')

            # Filter to only those with recent consent signatures
            newly_signed_from_assignments = []
            for assignment in signed_assignments:
                recent_consent = assignment.participant.consent_records.filter(
                    consent_form=assignment.consent_form,
                    status='signed',
                    signed_at__gte=timezone.now() - timedelta(days=7)
                ).first()
                if recent_consent:
                    newly_signed_from_assignments.append(assignment)

            # Limit to recent ones
            newly_signed_from_assignments = newly_signed_from_assignments[:5]

            # Get eligible participants for research (with valid consent)
            eligible_participants = ParticipantProfile.objects.filter(
                consent_records__status='signed',
                consent_records__expires_at__gt=timezone.now(),
                is_active=True
            ).annotate(
                consent_count=Count('consent_records', filter=Q(consent_records__status='signed')),
                latest_consent_date=Max('consent_records__signed_at')
            ).distinct().order_by('-latest_consent_date')

            # Get participants ready for specific research types
            eeg_ready_participants = eligible_participants.filter(
                consent_records__consent_form__form_type='eeg',
                consent_records__status='signed'
            ).distinct()

            context.update({
                'researcher_profile': researcher_profile,

                # Basic statistics
                'total_participants': ParticipantProfile.objects.filter(is_active=True).count(),
                'consented_participants_count': consented_participants.count(),
                'eligible_participants_count': eligible_participants.count(),
                'pending_assignments_count': pending_assignments.count(),
                'signed_assignments_count': signed_assignments.count(),

                # Consent-related data
                'recent_assignments': recent_assignments,
                'pending_assignments': pending_assignments[:5],
                'signed_assignments': signed_assignments[:5],
                'newly_signed_consents': newly_signed_from_assignments[:5],
                'recent_signed_consents': recent_signed_consents[:5],

                # Participant eligibility data
                'eligible_participants': eligible_participants[:10],
                'consented_participants': consented_participants[:10],
                'eeg_ready_participants': eeg_ready_participants[:5],

                # Research participation data
                'pending_requests': ResearchParticipationRequest.objects.filter(
                    researcher=researcher_profile, status='pending'
                ).count(),
                'approved_requests': ResearchParticipationRequest.objects.filter(
                    researcher=researcher_profile, status='approved'
                ).count(),
                'recent_requests': ResearchParticipationRequest.objects.filter(
                    researcher=researcher_profile
                ).order_by('-created_at')[:5],
                'upcoming_sessions': ParticipantSession.objects.filter(
                    researcher=researcher_profile,
                    scheduled_date__gte=timezone.now(),
                    status__in=['scheduled', 'confirmed']
                ).order_by('scheduled_date')[:5],
            })
        except ResearcherProfile.DoesNotExist:
            # Create researcher profile if it doesn't exist
            researcher_profile = ResearcherProfile.objects.create(user=request.user)
            context['researcher_profile'] = researcher_profile

        return render(request, 'user_management/researcher_dashboard.html', context)

    elif user_profile.is_regular_user:
        # Participant dashboard
        try:
            participant_profile = request.user.participant_profile
            # Get consent assignments with detailed status
            pending_assignments = ConsentAssignment.objects.filter(
                participant=participant_profile,
                status__in=['assigned', 'sent']
            ).select_related('consent_form', 'researcher__user').order_by('-assigned_at')

            # Get all consent records with status information
            consent_records = ConsentRecord.objects.filter(
                participant=participant_profile
            ).select_related('consent_form').order_by('-created_at')

            # Separate pending and completed consents
            pending_consents = consent_records.filter(status='pending')
            signed_consents = consent_records.filter(status='signed')

            context.update({
                'participant_profile': participant_profile,
                'consent_records': consent_records,
                'pending_consents': pending_consents,
                'signed_consents': signed_consents,
                'pending_assignments': pending_assignments,
                'pending_assignments_count': pending_assignments.count(),
                # Research participation data
                'participation_requests': ResearchParticipationRequest.objects.filter(
                    participant=participant_profile
                ).order_by('-created_at')[:5],
                'pending_requests': ResearchParticipationRequest.objects.filter(
                    participant=participant_profile, status='pending'
                ).order_by('-created_at'),
                'pending_requests_count': ResearchParticipationRequest.objects.filter(
                    participant=participant_profile, status='pending'
                ).count(),
                'approved_requests': ResearchParticipationRequest.objects.filter(
                    participant=participant_profile, status='approved'
                ).count(),
                'upcoming_sessions': ParticipantSession.objects.filter(
                    participant=participant_profile,
                    scheduled_date__gte=timezone.now(),
                    status__in=['scheduled', 'confirmed']
                ).order_by('scheduled_date')[:3],
                # Profile completion status
                'profile_complete': participant_profile.is_profile_complete,
                'profile_completion_percentage': participant_profile.profile_completion_percentage,
            })

            # Get or create availability profile
            availability, created = ParticipantAvailability.objects.get_or_create(
                participant=participant_profile
            )
            context['availability'] = availability

        except ParticipantProfile.DoesNotExist:
            # Create participant profile if it doesn't exist
            participant_profile = ParticipantProfile.objects.create(user=request.user)
            context.update({
                'participant_profile': participant_profile,
                'consent_records': [],
                'pending_consents': [],
                'participation_requests': [],
                'pending_requests': [],
                'pending_requests_count': 0,
                'approved_requests': 0,
                'upcoming_sessions': [],
                'consent_assignments': [],
                # Profile completion status (new profile will be incomplete)
                'profile_complete': participant_profile.is_profile_complete,
                'profile_completion_percentage': participant_profile.profile_completion_percentage,
            })

        return render(request, 'user_management/participant_dashboard.html', context)

    else:
        # Guest user
        messages.info(request, "Please register to access participant features.")
        return redirect('home')

@login_required
def participant_list(request):
    """List all participants (researcher only)"""
    if not hasattr(request.user, 'profile') or not request.user.profile.is_trainer:
        messages.error(request, "Access denied. Researcher privileges required.")
        return redirect('user_management:dashboard')

    participants = ParticipantProfile.objects.filter(is_active=True).order_by('-date_registered')

    # Search functionality
    search_query = request.GET.get('search', '')
    if search_query:
        participants = participants.filter(
            Q(user__username__icontains=search_query) |
            Q(user__first_name__icontains=search_query) |
            Q(user__last_name__icontains=search_query) |
            Q(user__email__icontains=search_query)
        )

    # Pagination
    paginator = Paginator(participants, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'total_participants': participants.count(),
    }

    return render(request, 'user_management/participant_list.html', context)

@login_required
def participant_detail(request, participant_id):
    """View participant details (researcher only)"""
    if not hasattr(request.user, 'profile') or not request.user.profile.is_trainer:
        messages.error(request, "Access denied. Researcher privileges required.")
        return redirect('user_management:dashboard')

    participant = get_object_or_404(ParticipantProfile, participant_id=participant_id)
    consent_records = ConsentRecord.objects.filter(participant=participant).order_by('-created_at')

    context = {
        'participant': participant,
        'consent_records': consent_records,
        'valid_consents': consent_records.filter(status='signed', is_expired=False, is_withdrawn=False),
        'pending_consents': consent_records.filter(status='pending'),
    }

    return render(request, 'user_management/participant_detail.html', context)

@login_required
def consent_management(request):
    """Manage consent forms (researcher only)"""
    if not hasattr(request.user, 'profile') or not request.user.profile.is_trainer:
        messages.error(request, "Access denied. Researcher privileges required.")
        return redirect('user_management:dashboard')

    consent_forms = ConsentForm.objects.filter(is_active=True).order_by('-created_at')

    context = {
        'consent_forms': consent_forms,
        'total_forms': consent_forms.count(),
    }

    return render(request, 'user_management/consent_management.html', context)

@login_required
def create_consent_form(request):
    """Create a new consent form (researcher only)"""
    # Check if user has trainer privileges
    if not hasattr(request.user, 'profile') or not request.user.profile.is_trainer:
        messages.error(request, "Access denied. Researcher privileges required.")
        return redirect('user_management:dashboard')

    if request.method == 'POST':
        form = ConsentFormForm(request.POST)
        if form.is_valid():
            try:
                consent_form = form.save(commit=False)
                consent_form.created_by = request.user
                consent_form.save()

                messages.success(
                    request,
                    f"Consent form '{consent_form.title}' (v{consent_form.version}) created successfully!"
                )
                return redirect('user_management:consent_management')

            except Exception as e:
                messages.error(
                    request,
                    f"Error creating consent form: {str(e)}. Please check your input and try again."
                )
        else:
            # Add form errors to messages for better user feedback
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field.title()}: {error}")
    else:
        form = ConsentFormForm()

    context = {
        'form': form,
    }
    return render(request, 'user_management/create_consent_form.html', context)

@login_required
def edit_consent_form(request, form_id):
    """Edit an existing consent form (researcher only)"""
    if not hasattr(request.user, 'profile') or not request.user.profile.is_trainer:
        messages.error(request, "Access denied. Researcher privileges required.")
        return redirect('user_management:dashboard')

    consent_form = get_object_or_404(ConsentForm, id=form_id)

    if request.method == 'POST':
        form = ConsentFormForm(request.POST, instance=consent_form)
        if form.is_valid():
            form.save()
            messages.success(request, f"Consent form '{consent_form.title}' updated successfully.")
            return redirect('user_management:consent_management')
    else:
        form = ConsentFormForm(instance=consent_form)

    return render(request, 'user_management/edit_consent_form.html', {
        'form': form,
        'consent_form': consent_form
    })

@login_required
def assign_consent_form(request, form_id=None):
    """Assign a consent form to participants (researcher only)"""
    if not hasattr(request.user, 'researcher_profile'):
        messages.error(request, "Researcher profile required.")
        return redirect('user_management:dashboard')

    researcher_profile = request.user.researcher_profile
    consent_form = None

    # If form_id is provided, get the specific consent form
    if form_id:
        consent_form = get_object_or_404(ConsentForm, id=form_id)

    if request.method == 'POST':
        # If we have a pre-selected consent form, ensure it's in the POST data
        post_data = request.POST.copy()
        if consent_form and 'consent_form' not in post_data:
            post_data['consent_form'] = consent_form.id

        form = ConsentAssignmentForm(
            post_data,
            researcher=researcher_profile,
            consent_form=consent_form
        )
        if form.is_valid():
            try:
                assignment = form.save(commit=False)
                assignment.researcher = researcher_profile

                # If we have a specific consent form, use it
                if consent_form:
                    assignment.consent_form = consent_form

                assignment.save()

                # Create consent record
                consent_record, created = ConsentRecord.objects.get_or_create(
                    participant=assignment.participant,
                    consent_form=assignment.consent_form,
                    defaults={
                        'status': 'pending',
                        'expires_at': timezone.now() + timedelta(days=assignment.consent_form.expiration_period)
                    }
                )

                # Send email notification to participant
                email_sent = False
                email_error = None
                try:
                    notify_consent_assignment(assignment)
                    email_sent = True
                    messages.success(request, "Email notification sent to participant.")
                except Exception as e:
                    email_error = str(e)
                    messages.warning(request, f"Assignment successful, but email notification failed: {email_error}")

                # Log the action
                ConsentAuditLog.log_action(
                    user=request.user,
                    action='form_assigned',
                    consent_record=consent_record,
                    consent_form=assignment.consent_form,
                    description=f"Consent form assigned to {assignment.participant.user.username}",
                    metadata={
                        'assignment_id': assignment.id,
                        'form_id': assignment.consent_form.id,
                        'participant_id': assignment.participant.id,
                        'email_sent': email_sent,
                        'email_error': email_error,
                    }
                )

                messages.success(request, f"Consent form '{assignment.consent_form.title}' successfully assigned to {assignment.participant.user.username}")
                return redirect('user_management:consent_management')

            except Exception as e:
                messages.error(request, f"Error assigning consent form: {str(e)}")
        else:
            # Add form errors to messages for better user feedback
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field.replace('_', ' ').title()}: {error}")
    else:
        # Initialize form with pre-selected consent form if provided
        initial_data = {}
        if consent_form:
            initial_data['consent_form'] = consent_form
        form = ConsentAssignmentForm(
            researcher=researcher_profile,
            consent_form=consent_form,
            initial=initial_data
        )

    context = {
        'form': form,
        'consent_form': consent_form,
        'researcher_profile': researcher_profile,
    }
    return render(request, 'user_management/assign_consent_form.html', context)

@login_required
def view_consent_form(request, form_id):
    """View a consent form (for participants and researchers)"""
    consent_form = get_object_or_404(ConsentForm, id=form_id, is_active=True)

    # Check if user has permission to view this consent form
    can_view = False

    if hasattr(request.user, 'participant_profile'):
        # Participants can view forms assigned to them
        participant = request.user.participant_profile
        can_view = ConsentRecord.objects.filter(
            participant=participant,
            consent_form=consent_form
        ).exists()
    elif hasattr(request.user, 'researcher_profile'):
        # Researchers can view all forms
        can_view = True
    elif hasattr(request.user, 'profile') and request.user.profile.is_trainer:
        # Trainers can view all forms
        can_view = True

    if not can_view:
        messages.error(request, "You don't have permission to view this consent form.")
        return redirect('user_management:dashboard')

    context = {
        'consent_form': consent_form,
    }
    return render(request, 'user_management/view_consent_form.html', context)

@login_required
def download_consent_pdf(request, record_id):
    """Download consent form as PDF"""
    consent_record = get_object_or_404(ConsentRecord, id=record_id)

    # Check if user has permission to download this PDF
    can_download = False

    if hasattr(request.user, 'participant_profile'):
        # Participants can download their own consent records
        can_download = consent_record.participant == request.user.participant_profile
    elif hasattr(request.user, 'researcher_profile'):
        # Researchers can download all PDFs
        can_download = True
    elif hasattr(request.user, 'profile') and request.user.profile.is_trainer:
        # Trainers can download all PDFs
        can_download = True

    if not can_download:
        messages.error(request, "You don't have permission to download this consent form.")
        return redirect('user_management:dashboard')

    # For now, return a simple response - you can implement PDF generation later
    from django.http import HttpResponse
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="consent_form_{consent_record.id}.pdf"'

    # TODO: Implement actual PDF generation
    # For now, return a placeholder message
    response.write(b"PDF generation not yet implemented. This is a placeholder.")

    return response

@login_required
def sign_consent(request, record_id):
    """Sign a consent form with comprehensive validation (participant only)"""
    consent_record = get_object_or_404(ConsentRecord, id=record_id)

    # Check if user is the participant
    if hasattr(request.user, 'participant_profile'):
        if consent_record.participant != request.user.participant_profile:
            messages.error(request, "Access denied. You can only sign your own consent forms.")
            return redirect('user_management:dashboard')
    elif not (hasattr(request.user, 'profile') and request.user.profile.is_trainer):
        messages.error(request, "Access denied.")
        return redirect('user_management:dashboard')

    # Check if consent is already signed
    if consent_record.status == 'signed':
        messages.info(request, "This consent form has already been signed.")
        return redirect('user_management:dashboard')

    # Get related assignment for additional context
    assignment = ConsentAssignment.objects.filter(
        participant=consent_record.participant,
        consent_form=consent_record.consent_form
    ).first()

    if request.method == 'POST':
        form = ConsentSigningForm(request.POST, consent_record=consent_record)
        if form.is_valid():
            try:
                # Update consent record
                consent_record.status = 'signed'
                consent_record.signed_at = timezone.now()
                consent_record.ip_address = request.META.get('REMOTE_ADDR')
                consent_record.user_agent = request.META.get('HTTP_USER_AGENT', '')

                # Store digital signature and acknowledgments
                consent_record.signature_data = form.cleaned_data['digital_signature']

                # Store specific consents based on form type
                if 'eeg_recording_consent' in form.cleaned_data:
                    consent_record.agreed_to_eeg_recording = form.cleaned_data['eeg_recording_consent']
                if 'audio_recording_consent' in form.cleaned_data:
                    consent_record.agreed_to_audio_recording = form.cleaned_data['audio_recording_consent']
                if 'data_sharing_consent' in form.cleaned_data:
                    consent_record.agreed_to_data_sharing = form.cleaned_data['data_sharing_consent']

                consent_record.save()

                # Update assignment status if exists
                if assignment:
                    assignment.status = 'signed'
                    assignment.save()

                # Create audit log entry
                ConsentAuditLog.log_action(
                    user=request.user,
                    action='consent_signed',
                    consent_record=consent_record,
                    consent_form=consent_record.consent_form,
                    description=f"Consent form signed by {request.user.username}",
                    metadata={
                        'ip_address': request.META.get('REMOTE_ADDR'),
                        'user_agent': request.META.get('HTTP_USER_AGENT', ''),
                        'signature': form.cleaned_data['digital_signature'],
                        'assignment_id': assignment.id if assignment else None,
                    }
                )

                # Send confirmation emails
                try:
                    from .email_notifications import notify_consent_signed
                    notify_consent_signed(consent_record, assignment)
                except Exception as e:
                    # Log error but don't fail the signing process
                    print(f"Failed to send confirmation email: {e}")

                messages.success(request,
                    f"Consent form '{consent_record.consent_form.title}' signed successfully! "
                    f"You will receive a confirmation email shortly.")
                return redirect('user_management:dashboard')

            except Exception as e:
                messages.error(request, f"Error signing consent form: {str(e)}")
    else:
        form = ConsentSigningForm(consent_record=consent_record)

    context = {
        'consent_record': consent_record,
        'assignment': assignment,
        'form': form,
        'consent_form': consent_record.consent_form,
    }
    return render(request, 'user_management/sign_consent.html', context)

@login_required
def review_consent_assignment(request, assignment_id):
    """Review a consent assignment before signing (participant only)"""
    assignment = get_object_or_404(ConsentAssignment, id=assignment_id)

    # Check if user is the participant
    if hasattr(request.user, 'participant_profile'):
        if assignment.participant != request.user.participant_profile:
            messages.error(request, "Access denied. You can only review your own consent assignments.")
            return redirect('user_management:dashboard')
    else:
        messages.error(request, "Access denied.")
        return redirect('user_management:dashboard')

    # Mark assignment as viewed
    if assignment.status == 'assigned':
        assignment.status = 'viewed'
        assignment.save()

    # Get or create consent record
    consent_record, created = ConsentRecord.objects.get_or_create(
        participant=assignment.participant,
        consent_form=assignment.consent_form,
        defaults={
            'status': 'pending',
            'expires_at': timezone.now() + timedelta(days=assignment.consent_form.expiration_period)
        }
    )

    context = {
        'assignment': assignment,
        'consent_record': consent_record,
        'consent_form': assignment.consent_form,
        'researcher': assignment.researcher,
    }
    return render(request, 'user_management/review_consent_assignment.html', context)

@login_required
def withdraw_consent(request, record_id):
    """Withdraw consent (participant only)"""
    consent_record = get_object_or_404(ConsentRecord, id=record_id)

    # Check if user is the participant
    if hasattr(request.user, 'participant_profile'):
        if consent_record.participant != request.user.participant_profile:
            messages.error(request, "Access denied. You can only withdraw your own consent.")
            return redirect('user_management:dashboard')
    else:
        messages.error(request, "Access denied.")
        return redirect('user_management:dashboard')

    if request.method == 'POST':
        withdrawal_reason = request.POST.get('withdrawal_reason', '')
        consent_record.status = 'withdrawn'
        consent_record.is_withdrawn = True
        consent_record.withdrawal_date = timezone.now()
        consent_record.withdrawal_reason = withdrawal_reason
        consent_record.save()

        messages.success(request, "Consent withdrawn successfully.")
        return redirect('user_management:dashboard')

    return render(request, 'user_management/withdraw_consent.html', {
        'consent_record': consent_record
    })

@login_required
def edit_profile(request):
    """Edit user profile"""
    user_profile = getattr(request.user, 'profile', None)

    if not user_profile:
        messages.error(request, "User profile not found.")
        return redirect('home')

    if user_profile.is_trainer:
        # Researcher profile
        try:
            researcher_profile = request.user.researcher_profile
        except ResearcherProfile.DoesNotExist:
            researcher_profile = ResearcherProfile.objects.create(user=request.user)

        if request.method == 'POST':
            form = ResearcherProfileForm(request.POST, instance=researcher_profile)
            if form.is_valid():
                form.save()
                messages.success(request, "Profile updated successfully.")
                return redirect('user_management:dashboard')
        else:
            form = ResearcherProfileForm(instance=researcher_profile)

        return render(request, 'user_management/edit_researcher_profile.html', {'form': form})

    elif user_profile.is_regular_user:
        # Participant profile
        try:
            participant_profile = request.user.participant_profile
        except ParticipantProfile.DoesNotExist:
            participant_profile = ParticipantProfile.objects.create(user=request.user)

        if request.method == 'POST':
            form = ParticipantProfileForm(request.POST, instance=participant_profile)
            if form.is_valid():
                form.save()
                messages.success(request, "✅ Profile updated successfully! Your information has been saved.")
                return redirect('user_management:dashboard')
            else:
                messages.error(request, "Please correct the errors below.")
        else:
            form = ParticipantProfileForm(instance=participant_profile)

        context = {
            'form': form,
            'participant_profile': participant_profile,
            'consent_forms': []  # Add empty list for now
        }
        return render(request, 'user_management/edit_participant_profile.html', context)

    else:
        messages.error(request, "Profile editing not available for guest users.")
        return redirect('home')

# API Views
@login_required
def participant_list_api(request):
    """API endpoint for participant list"""
    if not hasattr(request.user, 'profile') or not request.user.profile.is_trainer:
        return JsonResponse({'error': 'Access denied'}, status=403)

    participants = ParticipantProfile.objects.filter(is_active=True).values(
        'participant_id', 'user__username', 'user__first_name', 'user__last_name',
        'gender', 'has_valid_consent', 'date_registered'
    )

    return JsonResponse({'participants': list(participants)})

@login_required
def consent_status_api(request):
    """API endpoint for consent status"""
    if hasattr(request.user, 'participant_profile'):
        participant = request.user.participant_profile
        consent_records = ConsentRecord.objects.filter(participant=participant).values(
            'consent_form__title', 'status', 'signed_at', 'expires_at'
        )
        return JsonResponse({'consent_records': list(consent_records)})

    return JsonResponse({'error': 'Not a participant'}, status=400)


@login_required
def eligible_participants_view(request):
    """View for researchers to see participants eligible for research"""
    if not hasattr(request.user, 'researcher_profile'):
        messages.error(request, "Access denied. Researcher profile required.")
        return redirect('user_management:dashboard')

    researcher_profile = request.user.researcher_profile

    # Get participants with active consent forms
    eligible_participants = ParticipantProfile.objects.filter(
        consent_records__status='signed',
        consent_records__expires_at__gt=timezone.now(),
        is_active=True
    ).annotate(
        consent_count=Count('consent_records', filter=Q(consent_records__status='signed')),
        latest_consent_date=Max('consent_records__signed_at'),
        eeg_consent=Count('consent_records', filter=Q(
            consent_records__status='signed',
            consent_records__consent_form__form_type='eeg'
        )),
        audio_consent=Count('consent_records', filter=Q(
            consent_records__status='signed',
            consent_records__consent_form__form_type='audio'
        )),
        data_sharing_consent=Count('consent_records', filter=Q(
            consent_records__status='signed',
            consent_records__consent_form__form_type='data_sharing'
        ))
    ).distinct().order_by('-latest_consent_date')

    # Filter by research type if specified
    research_type = request.GET.get('type')
    if research_type:
        if research_type == 'eeg':
            eligible_participants = eligible_participants.filter(eeg_consent__gt=0)
        elif research_type == 'audio':
            eligible_participants = eligible_participants.filter(audio_consent__gt=0)
        elif research_type == 'data_sharing':
            eligible_participants = eligible_participants.filter(data_sharing_consent__gt=0)

    # Pagination
    paginator = Paginator(eligible_participants, 20)
    page_number = request.GET.get('page')
    participants = paginator.get_page(page_number)

    context = {
        'participants': participants,
        'research_type': research_type,
        'total_eligible': eligible_participants.count(),
    }

    return render(request, 'user_management/eligible_participants.html', context)


@login_required
def invite_participant_to_study(request, participant_id):
    """Invite a consented participant to a specific research study"""
    if not hasattr(request.user, 'researcher_profile'):
        messages.error(request, "Access denied. Researcher profile required.")
        return redirect('user_management:dashboard')

    researcher_profile = request.user.researcher_profile
    participant = get_object_or_404(ParticipantProfile, participant_id=participant_id)

    # Check if participant has valid consent
    has_valid_consent = participant.consent_records.filter(
        status='signed',
        expires_at__gt=timezone.now()
    ).exists()

    if not has_valid_consent:
        messages.error(request, "Participant does not have valid consent forms.")
        return redirect('user_management:eligible_participants_view')

    if request.method == 'POST':
        # Create research participation request
        study_title = request.POST.get('study_title')
        study_description = request.POST.get('study_description')
        session_duration = request.POST.get('session_duration')
        compensation = request.POST.get('compensation')
        contact_method = request.POST.get('contact_method', 'email')

        participation_request = ResearchParticipationRequest.objects.create(
            participant=participant,
            researcher=researcher_profile,
            request_type='specific_researcher',  # This is a researcher-initiated invitation
            study_title=study_title,
            study_description=study_description,
            session_duration=session_duration,
            compensation=compensation,
            contact_method=contact_method,
            status='pending',
            expires_at=timezone.now() + timedelta(days=14)  # 14-day expiration
        )

        # Send notification email
        try:
            notify_participation_request(participation_request)
            messages.success(request, f"Study invitation sent to {participant.user.get_full_name() or participant.user.username}")
        except Exception as e:
            messages.warning(request, f"Invitation created but email notification failed: {str(e)}")

        return redirect('user_management:eligible_participants_view')

    context = {
        'participant': participant,
        'researcher': researcher_profile,
    }

    return render(request, 'user_management/invite_participant.html', context)


# Research Participation Views

@login_required
def join_research_study(request):
    """Entry point for users wanting to join research studies"""
    if not hasattr(request.user, 'participant_profile'):
        messages.error(request, "Participant profile required.")
        return redirect('user_management:dashboard')

    participant_profile = request.user.participant_profile

    # Get or create availability profile
    availability, created = ParticipantAvailability.objects.get_or_create(
        participant=participant_profile
    )

    # Get available researchers
    researchers = ResearcherProfile.objects.filter(is_active=True)

    # Get pending requests
    pending_requests = ResearchParticipationRequest.objects.filter(
        participant=participant_profile,
        status='pending'
    )

    context = {
        'participant_profile': participant_profile,
        'availability': availability,
        'researchers': researchers,
        'pending_requests': pending_requests,
        'pending_requests_count': pending_requests.count(),
    }
    return render(request, 'user_management/join_research_study.html', context)


@login_required
def submit_participation_request(request):
    """Handle AJAX submission of participation requests"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Invalid request method'})

    if not hasattr(request.user, 'participant_profile'):
        return JsonResponse({'success': False, 'error': 'Participant profile required'})

    participant_profile = request.user.participant_profile
    request_type = request.POST.get('request_type')
    researcher_id = request.POST.get('researcher_id')
    message = request.POST.get('message', '')

    try:
        # Check for existing pending requests
        existing_request = ResearchParticipationRequest.objects.filter(
            participant=participant_profile,
            status='pending'
        ).first()

        if existing_request:
            return JsonResponse({
                'success': False,
                'error': 'You already have a pending participation request'
            })

        # Create the request
        participation_request = ResearchParticipationRequest.objects.create(
            participant=participant_profile,
            request_type=request_type,
            participant_message=message,
            status='pending',
            expires_at=timezone.now() + timedelta(days=14)
        )

        if request_type == 'specific_researcher' and researcher_id:
            try:
                researcher = ResearcherProfile.objects.get(id=researcher_id)
                participation_request.researcher = researcher
                participation_request.save()
            except ResearcherProfile.DoesNotExist:
                return JsonResponse({'success': False, 'error': 'Invalid researcher selected'})

        # Update availability for open invitations
        if request_type == 'open_invitation':
            availability, created = ParticipantAvailability.objects.get_or_create(
                participant=participant_profile
            )
            availability.is_open_to_invitations = True
            availability.save()

        # Send email notification
        try:
            notify_participation_request(participation_request)
        except Exception as e:
            print(f"Failed to send email notification: {e}")

        return JsonResponse({
            'success': True,
            'message': 'Your participation request has been submitted successfully!'
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def confirm_participation(request, request_id):
    """Handle participant confirmation of research invitations"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Invalid request method'})

    if not hasattr(request.user, 'participant_profile'):
        return JsonResponse({'success': False, 'error': 'Participant profile required'})

    try:
        participation_request = ResearchParticipationRequest.objects.get(
            id=request_id,
            participant=request.user.participant_profile
        )

        action = request.POST.get('action')

        if action == 'confirm':
            participation_request.status = 'approved'
            participation_request.participant_response_date = timezone.now()
            participation_request.save()

            # Send notification to researcher
            try:
                notify_request_response(participation_request)
            except Exception as e:
                print(f"Failed to send email notification: {e}")

            return JsonResponse({
                'success': True,
                'message': 'Research invitation confirmed successfully!'
            })

        elif action == 'decline':
            participation_request.status = 'declined'
            participation_request.participant_response_date = timezone.now()
            participation_request.save()

            return JsonResponse({
                'success': True,
                'message': 'Research invitation declined.'
            })

        else:
            return JsonResponse({'success': False, 'error': 'Invalid action'})

    except ResearchParticipationRequest.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Request not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def create_participation_request(request):
    """Create a new research participation request"""
    if not hasattr(request.user, 'participant_profile'):
        messages.error(request, "Participant profile required.")
        return redirect('user_management:dashboard')

    participant_profile = request.user.participant_profile

    if request.method == 'POST':
        form = ResearchParticipationRequestForm(request.POST, user=request.user)
        if form.is_valid():
            participation_request = form.save(commit=False)
            participation_request.participant = participant_profile

            # Set researcher to None for open invitations
            if form.cleaned_data['request_type'] == 'open_invitation':
                participation_request.researcher = None

            participation_request.save()

            # Send email notification
            try:
                notify_participation_request(participation_request)
            except Exception as e:
                # Log error but don't fail the request
                print(f"Failed to send email notification: {e}")

            # Log the action
            ConsentAuditLog.log_action(
                user=request.user,
                action='data_accessed',  # Using closest available action
                description=f"Research participation request created: {participation_request.request_type}",
                metadata={
                    'request_type': participation_request.request_type,
                    'researcher': participation_request.researcher.user.username if participation_request.researcher else None,
                }
            )

            messages.success(request, "Your participation request has been submitted successfully!")
            return redirect('user_management:dashboard')
    else:
        form = ResearchParticipationRequestForm(user=request.user)

    context = {
        'form': form,
        'participant_profile': participant_profile,
    }
    return render(request, 'user_management/create_participation_request.html', context)


@login_required
def participation_requests(request):
    """View all participation requests for current user"""
    if not hasattr(request.user, 'participant_profile'):
        messages.error(request, "Participant profile required.")
        return redirect('user_management:dashboard')

    participant_profile = request.user.participant_profile
    requests = ResearchParticipationRequest.objects.filter(
        participant=participant_profile
    ).order_by('-created_at')

    # Pagination
    paginator = Paginator(requests, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'participant_profile': participant_profile,
    }
    return render(request, 'user_management/participation_requests.html', context)


@login_required
def researcher_requests(request):
    """View participation requests for researchers"""
    if not hasattr(request.user, 'researcher_profile'):
        messages.error(request, "Researcher profile required.")
        return redirect('user_management:dashboard')

    researcher_profile = request.user.researcher_profile

    # Get requests directed to this researcher or open invitations
    requests = ResearchParticipationRequest.objects.filter(
        Q(researcher=researcher_profile) | Q(researcher=None, request_type='open_invitation')
    ).order_by('-created_at')

    # Filter by status if requested
    status_filter = request.GET.get('status')
    if status_filter:
        requests = requests.filter(status=status_filter)

    # Pagination
    paginator = Paginator(requests, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'researcher_profile': researcher_profile,
        'status_filter': status_filter,
        'status_choices': ResearchParticipationRequest.STATUS_CHOICES,
    }
    return render(request, 'user_management/researcher_requests.html', context)


@login_required
def respond_to_request(request, request_id):
    """Researcher response to participation request"""
    if not hasattr(request.user, 'researcher_profile'):
        messages.error(request, "Researcher profile required.")
        return redirect('user_management:dashboard')

    researcher_profile = request.user.researcher_profile
    participation_request = get_object_or_404(
        ResearchParticipationRequest,
        id=request_id,
        status='pending'
    )

    # Check if researcher can respond to this request
    if participation_request.researcher and participation_request.researcher != researcher_profile:
        messages.error(request, "You cannot respond to this request.")
        return redirect('user_management:researcher_requests')

    if request.method == 'POST':
        form = ResearcherResponseForm(request.POST)
        if form.is_valid():
            action = form.cleaned_data['action']
            response_message = form.cleaned_data['researcher_response']

            participation_request.researcher = researcher_profile
            participation_request.researcher_response = response_message
            participation_request.responded_at = timezone.now()

            if action == 'approve':
                participation_request.status = 'approved'
                messages.success(request, "Request approved successfully!")
            else:
                participation_request.status = 'declined'
                messages.success(request, "Request declined.")

            participation_request.save()

            # Send email notification to participant
            try:
                notify_request_response(participation_request)
            except Exception as e:
                # Log error but don't fail the response
                print(f"Failed to send email notification: {e}")

            # Log the action
            ConsentAuditLog.log_action(
                user=request.user,
                action='data_accessed',
                description=f"Research participation request {action}d",
                metadata={
                    'request_id': participation_request.id,
                    'action': action,
                    'participant': participation_request.participant.user.username,
                }
            )

            return redirect('user_management:researcher_requests')
    else:
        form = ResearcherResponseForm()

    context = {
        'form': form,
        'participation_request': participation_request,
        'researcher_profile': researcher_profile,
    }
    return render(request, 'user_management/respond_to_request.html', context)


@login_required
def manage_availability(request):
    """Manage participant availability"""
    if not hasattr(request.user, 'participant_profile'):
        messages.error(request, "Participant profile required.")
        return redirect('user_management:dashboard')

    participant_profile = request.user.participant_profile
    availability, created = ParticipantAvailability.objects.get_or_create(
        participant=participant_profile
    )

    if request.method == 'POST':
        form = ParticipantAvailabilityForm(request.POST, instance=availability)
        if form.is_valid():
            form.save()
            messages.success(request, "Availability updated successfully!")
            return redirect('user_management:dashboard')
    else:
        form = ParticipantAvailabilityForm(instance=availability)

    context = {
        'form': form,
        'availability': availability,
        'participant_profile': participant_profile,
    }
    return render(request, 'user_management/manage_availability.html', context)





@login_required
def schedule_session(request):
    """Schedule a research session (researcher only)"""
    if not hasattr(request.user, 'researcher_profile'):
        messages.error(request, "Researcher profile required.")
        return redirect('user_management:dashboard')

    researcher_profile = request.user.researcher_profile

    if request.method == 'POST':
        form = ParticipantSessionForm(request.POST, researcher=researcher_profile)
        if form.is_valid():
            session = form.save(commit=False)
            session.researcher = researcher_profile
            session.save()

            messages.success(request, f"Session '{session.title}' scheduled successfully!")
            return redirect('user_management:dashboard')
    else:
        form = ParticipantSessionForm(researcher=researcher_profile)

    context = {
        'form': form,
        'researcher_profile': researcher_profile,
    }
    return render(request, 'user_management/schedule_session.html', context)


@login_required
def participant_sessions(request):
    """View participant's scheduled sessions"""
    if not hasattr(request.user, 'participant_profile'):
        messages.error(request, "Participant profile required.")
        return redirect('user_management:dashboard')

    participant_profile = request.user.participant_profile
    sessions = ParticipantSession.objects.filter(
        participant=participant_profile
    ).order_by('-scheduled_date')

    # Filter by status if requested
    status_filter = request.GET.get('status')
    if status_filter:
        sessions = sessions.filter(status=status_filter)

    # Pagination
    paginator = Paginator(sessions, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'participant_profile': participant_profile,
        'status_filter': status_filter,
        'status_choices': ParticipantSession.STATUS_CHOICES,
    }
    return render(request, 'user_management/participant_sessions.html', context)


@login_required
def edit_researcher_profile(request):
    """Edit researcher profile"""
    if not hasattr(request.user, 'researcher_profile'):
        messages.error(request, "Researcher profile required.")
        return redirect('user_management:dashboard')

    researcher_profile = request.user.researcher_profile

    if request.method == 'POST':
        form = ResearcherProfileForm(request.POST, instance=researcher_profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Researcher profile updated successfully!")
            return redirect('user_management:dashboard')
    else:
        form = ResearcherProfileForm(instance=researcher_profile)

    context = {
        'form': form,
        'researcher_profile': researcher_profile,
    }
    return render(request, 'user_management/edit_researcher_profile.html', context)


@login_required
def workflow_guide(request):
    """Display workflow guide for users"""
    context = {
        'user_profile': request.user.profile,
    }
    return render(request, 'user_management/workflow_guide.html', context)
