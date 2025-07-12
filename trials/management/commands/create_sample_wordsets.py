# trials/management/commands/create_sample_wordsets.py
# Django management command to create sample word sets for visual trials
from django.core.management.base import BaseCommand
from trials.models import WordSet, WordSetItem

class Command(BaseCommand):
    help = 'Create sample word sets for visual trials'

    def handle(self, *args, **options):
        # Sample word sets to create
        word_sets = [
            {
                'name': 'Basic Actions',
                'description': 'Common action words for motor imagery',
                'words': ['run', 'walk', 'jump', 'sit', 'stand']
            },
            {
                'name': 'Hand Movements',
                'description': 'Hand and finger movements',
                'words': ['grasp', 'point', 'wave', 'clap', 'touch']
            },
            {
                'name': 'Body Parts',
                'description': 'Body part words',
                'words': ['hand', 'foot', 'head', 'arm', 'leg']
            },
            {
                'name': 'Emotions',
                'description': 'Emotional states',
                'words': ['happy', 'sad', 'angry', 'calm', 'excited']
            },
            {
                'name': 'Colors',
                'description': 'Basic color words',
                'words': ['red', 'blue', 'green', 'yellow', 'purple']
            },
            {
                'name': 'Motor Imagery',
                'description': 'Val Collection trial',
                'words': ['SQUEEZE', 'KICK', 'SPIN', 'BRIGHT', 'SPEAK']
            },
            {
                'name': 'Speller Vocabulary',
                'description': 'P300 words that can be spelled with current letters (A,E,I,O,S,H,L,N,P,R)',
                'words': [
                    'HE', 'SHE', 'HI', 'SO', 'IS', 'OR', 'NO', 'HELP', 'STOP',
                    'OPEN'
                ]
            }
        ]
       
        created_count = 0
       
        for word_set_data in word_sets:
            # Check if word set already exists
            if WordSet.objects.filter(name=word_set_data['name']).exists():
                self.stdout.write(
                    self.style.WARNING(f'Word set "{word_set_data["name"]}" already exists, skipping...')
                )
                continue
           
            # Create word set
            word_set = WordSet.objects.create(
                name=word_set_data['name'],
                description=word_set_data['description'],
                is_active=True
            )
           
            # Add words to the set
            for i, word in enumerate(word_set_data['words']):
                WordSetItem.objects.create(
                    word_set=word_set,
                    word=word,
                    order=i
                )
           
            created_count += 1
            self.stdout.write(
                self.style.SUCCESS(
                    f'Created word set "{word_set.name}" with {len(word_set_data["words"])} words'
                )
            )
       
        if created_count > 0:
            self.stdout.write(
                self.style.SUCCESS(f'\nSuccessfully created {created_count} word sets!')
            )
            self.stdout.write(
                'You can now use these word sets for visual trials, or create your own in the Django admin.'
            )
        else:
            self.stdout.write(
                self.style.WARNING('No new word sets were created (all already exist).')
            )