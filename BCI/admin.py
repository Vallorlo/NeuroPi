# motor_imagery/admin.py

from django.contrib import admin
from django.utils.html import format_html
from .models import EEGSession, TrainingModel, TrainingJob, Prediction, PredictionSession, ClassMetrics

@admin.register(EEGSession)
class EEGSessionAdmin(admin.ModelAdmin):
    list_display = ['session_name', 'user', 'created_at', 'duration', 'sampling_rate', 'file_size']
    list_filter = ['created_at', 'user', 'sampling_rate']
    search_fields = ['session_name', 'user__username']
    readonly_fields = ['created_at', 'duration']
    
    def file_size(self, obj):
        if obj.file_path:
            size = obj.file_path.size
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size / 1024:.1f} KB"
            else:
                return f"{size / (1024 * 1024):.1f} MB"
        return "-"
    file_size.short_description = "File Size"

@admin.register(TrainingModel)
class TrainingModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'accuracy_display', 'created_at', 'is_active', 'status_badge']
    list_filter = ['is_active', 'created_at', 'user']
    search_fields = ['name', 'user__username']
    readonly_fields = ['created_at', 'accuracy', 'config']
    filter_horizontal = ['training_sessions']
    
    def accuracy_display(self, obj):
        if obj.accuracy:
            return f"{obj.accuracy:.2f}%"
        return "-"
    accuracy_display.short_description = "Accuracy"
    
    def status_badge(self, obj):
        if obj.is_active:
            return format_html('<span style="color: green;">●</span> Active')
        return format_html('<span style="color: red;">●</span> Inactive')
    status_badge.short_description = "Status"

@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'status_badge', 'progress_bar', 'created_at', 'duration']
    list_filter = ['status', 'created_at', 'user']
    search_fields = ['user__username']
    readonly_fields = ['created_at', 'started_at', 'completed_at', 'training_log', 'error_message']
    filter_horizontal = ['sessions']
    
    def status_badge(self, obj):
        colors = {
            'pending': 'orange',
            'running': 'blue',
            'completed': 'green',
            'failed': 'red'
        }
        color = colors.get(obj.status, 'gray')
        return format_html(
            '<span style="color: {};">●</span> {}',
            color,
            obj.get_status_display()
        )
    status_badge.short_description = "Status"
    
    def progress_bar(self, obj):
        return format_html(
            '<div style="width: 100px; background-color: #f0f0f0; border: 1px solid #ccc;">'
            '<div style="width: {}%; background-color: #4CAF50; height: 20px;"></div>'
            '</div>',
            obj.progress
        )
    progress_bar.short_description = "Progress"
    
    def duration(self, obj):
        if obj.started_at and obj.completed_at:
            duration = obj.completed_at - obj.started_at
            return str(duration).split('.')[0]  # Remove microseconds
        return "-"
    duration.short_description = "Duration"

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'user', 'predicted_class', 'confidence_display', 'model']
    list_filter = ['predicted_class', 'timestamp', 'user', 'model']
    search_fields = ['user__username', 'predicted_class']
    readonly_fields = ['timestamp', 'probabilities']
    date_hierarchy = 'timestamp'
    
    def confidence_display(self, obj):
        return f"{obj.confidence:.2%}"
    confidence_display.short_description = "Confidence"

@admin.register(PredictionSession)
class PredictionSessionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'model', 'started_at', 'duration_display', 'prediction_count', 'is_active_badge']
    list_filter = ['is_active', 'started_at', 'user', 'model']
    search_fields = ['user__username', 'model__name']
    readonly_fields = ['started_at', 'ended_at']
    
    def duration_display(self, obj):
        duration = obj.duration()
        if duration:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            return f"{minutes}m {seconds}s"
        return "Ongoing" if obj.is_active else "-"
    duration_display.short_description = "Duration"
    
    def prediction_count(self, obj):
        return obj.predictions.count()
    prediction_count.short_description = "Predictions"
    
    def is_active_badge(self, obj):
        if obj.is_active:
            return format_html('<span style="color: green;">●</span> Active')
        return format_html('<span style="color: gray;">●</span> Ended')
    is_active_badge.short_description = "Status"

@admin.register(ClassMetrics)
class ClassMetricsAdmin(admin.ModelAdmin):
    list_display = ['model', 'class_name', 'precision_display', 'recall_display', 'f1_display', 'support']
    list_filter = ['model', 'class_name']
    search_fields = ['model__name', 'class_name']
    
    def precision_display(self, obj):
        return f"{obj.precision:.3f}"
    precision_display.short_description = "Precision"
    
    def recall_display(self, obj):
        return f"{obj.recall:.3f}"
    recall_display.short_description = "Recall"
    
    def f1_display(self, obj):
        return f"{obj.f1_score:.3f}"
    f1_display.short_description = "F1-Score"

# Admin site customization
admin.site.site_header = "Motor Imagery BCI Admin"
admin.site.site_title = "Motor Imagery BCI"
admin.site.index_title = "Welcome to Motor Imagery BCI Administration"