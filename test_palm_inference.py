import unittest
import math
from main import get_euclidean_length, calculate_x_hand_ref, interpret_traits, TRAITS

class TestPalmInference(unittest.TestCase):

    def test_euclidean_length(self):
        """Test the distance calculation between two points."""
        # 3-4-5 triangle
        length = get_euclidean_length(0, 0, 3, 4)
        self.assertEqual(length, 5.0)
        
        # Horizontal line
        length = get_euclidean_length(10, 10, 20, 10)
        self.assertEqual(length, 10.0)

    def test_calculate_x_hand_ref_defaults(self):
        """Test that hand reference defaults to 200.0 if lines are missing."""
        # Empty list
        self.assertEqual(calculate_x_hand_ref([]), 200.0)
        
        # Only Life line (Head and Heart missing)
        lines = [{"name": "Life", "x1": 0, "y1": 0, "x2": 0, "y2": 0}]
        self.assertEqual(calculate_x_hand_ref(lines), 200.0)

    def test_calculate_x_hand_ref_logic(self):
        """Test dynamic calculation of hand width based on Head/Heart lines."""
        # Head line at x=100..200, Heart line at x=100..300
        # The function looks for max diff between start/end points of these two lines.
        lines = [
            {"name": "Head", "x1": 100, "y1": 500, "x2": 200, "y2": 500},
            {"name": "Heart", "x1": 100, "y1": 300, "x2": 300, "y2": 300}
        ]
        
        # Possible x-diffs:
        # |100 - 100| = 0
        # |100 - 300| = 200
        # |200 - 100| = 100
        # |200 - 300| = 100
        # Max is 200
        
        self.assertEqual(calculate_x_hand_ref(lines), 200.0)

    def test_interpret_traits_life_line(self):
        """Test if a long Life Line is interpreted correctly."""
        # Hand width = 100 (Implicitly via Head/Heart setup below)
        # Life line length = 80 (0.8 normalized) -> Should be "mid" (0.60 - 0.92)
        
        lines = [
            # Setup for Ref Width = 100
            {"name": "Head", "x1": 0, "y1": 0, "x2": 0, "y2": 0},
            {"name": "Heart", "x1": 0, "y1": 0, "x2": 100, "y2": 0},
            
            # Life Line: Length 80
            {"name": "Life", "x1": 0, "y1": 0, "x2": 0, "y2": 80}
        ]
        
        result = interpret_traits(lines)
        life_trait = next(item for item in result if "Life Line" in item["title"])
        
        # Check Length interpretation
        length_result = life_trait["traits"][0]["result"] # 0 is length trait
        
        # Based on THRESHOLDS["life"]["length"] = (0.60, 0.92)
        # 0.8 is between 0.60 and 0.92, so it should be "mid"
        expected_text = TRAITS["life"]["length"]["mid"]["result"]
        self.assertEqual(length_result, expected_text)

    def test_interpret_traits_heart_line_extra_long(self):
        """Test the unique 'extra_long' logic for Heart Line."""
        # Hand width = 100
        # Heart line length = 95 (0.95 normalized) -> Should be "extra_long" (> 0.92)
        
        lines = [
             # Setup for Ref Width = 100
            {"name": "Head", "x1": 0, "y1": 0, "x2": 0, "y2": 0},
            {"name": "Heart", "x1": 0, "y1": 0, "x2": 100, "y2": 0},
            
            # The Heart Line itself (duplicate entry for logic, but valid for test)
            # We need a separate entry to be processed as the subject line
            # Length 95
            {"name": "Heart", "x1": 0, "y1": 0, "x2": 95, "y2": 0} 
        ]
        
        # Note: interpret_traits deduplicates or processes all. 
        # Since we have multiple "Heart" entries, it processes all of them.
        # We need to find the one that corresponds to our test line (length 95).
        
        results = interpret_traits(lines)
        
        # Find the result that came from our 95-length line
        # The other heart line (width reference) was length 100 (1.0 normalized)
        # Both > 0.92, so both should be Extra Long.
        
        for item in results:
            if "Heart Line" in item["title"]:
                length_result = item["traits"][0]["result"]
                expected_text = TRAITS["heart"]["length"]["extra_long"]["result"]
                self.assertEqual(length_result, expected_text)

if __name__ == '__main__':
    unittest.main()
