"""
Monitoring and health check utilities
"""
import psutil
import threading
from django.utils import timezone
from django.db import connection
from .models import CommunicationSession, CommunicationEvent


class SystemMonitor:
    """Monitor system resources and BCI communication health"""
    
    @staticmethod
    def get_system_stats():
        """Get current system resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'active_threads': threading.active_count(),
        }
    
    @staticmethod
    def get_communication_stats():
        """Get BCI communication statistics"""
        active_sessions = CommunicationSession.objects.filter(is_active=True).count()
        
        # Recent events (last hour)
        one_hour_ago = timezone.now() - timezone.timedelta(hours=1)
        recent_events = CommunicationEvent.objects.filter(
            timestamp__gte=one_hour_ago
        ).count()
        
        return {
            'active_sessions': active_sessions,
            'recent_events': recent_events,
            'database_connections': len(connection.queries),
        }
    
    @staticmethod
    def health_check():
        """Comprehensive health check"""
        try:
            # Test database
            CommunicationSession.objects.count()
            db_status = 'healthy'
        except Exception as e:
            db_status = f'error: {e}'
        
        # Test system resources
        system_stats = SystemMonitor.get_system_stats()
        
        # Determine overall health
        health_status = 'healthy'
        if system_stats['cpu_percent'] > 90:
            health_status = 'warning'
        if system_stats['memory_percent'] > 90:
            health_status = 'critical'
        if db_status != 'healthy':
            health_status = 'critical'
        
        return {
            'status': health_status,
            'database': db_status,
            'system': system_stats,
            'communication': SystemMonitor.get_communication_stats(),
            'timestamp': timezone.now().isoformat()
        }
