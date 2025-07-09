class WordPredictor:
    """
    Auto-complete system for the 10-word vocabulary
    """
    
    def __init__(self, vocabulary_words):
        self.vocabulary = vocabulary_words
        # Word frequency ranking
        self.word_frequencies = {
            'YES': 10, 'NO': 10,
            'HELP': 8, 'HI': 8,
            'STOP': 6, 'SO': 6, 'TO': 6,
            'IS': 4, 'IT': 4, 'OR': 4
        }
    
    def get_suggestions(self, partial_word):
        """Get word suggestions for partial input"""
        if not partial_word:
            return []
        
        partial_upper = partial_word.upper()
        suggestions = []
        
        for word in self.vocabulary:
            if word.startswith(partial_upper):
                score = self._calculate_score(partial_word, word)
                suggestions.append({
                    'word': word,
                    'score': score,
                    'frequency': self.word_frequencies.get(word, 1)
                })
        
        # Sort by score (match quality) and frequency
        suggestions.sort(key=lambda x: (x['score'], x['frequency']), reverse=True)
        return suggestions
    
    def _calculate_score(self, partial, full_word):
        """Calculate match score for auto-complete"""
        if not partial:
            return 0.0
        
        # Exact prefix match
        if full_word.startswith(partial.upper()):
            # Higher score for longer matches
            return len(partial) / len(full_word)
        
        return 0.0
    
    def get_best_match(self, partial_word):
        """Get the best matching word"""
        suggestions = self.get_suggestions(partial_word)
        return suggestions[0] if suggestions else None