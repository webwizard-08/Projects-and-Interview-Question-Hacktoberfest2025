from typing import List, Dict
from collections import defaultdict, Counter
from datetime import datetime
import json

class PythonInterviewSolutions:
    """Collection of common Python interview question solutions"""
    
    @staticmethod
    def list_comprehension_vs_map() -> tuple:
        """
        Demonstrates list comprehension vs map/filter/lambda
        
        Returns:
            tuple: Results from both approaches
        """
        numbers = range(1, 11)
        
        # Using list comprehension
        squares_comp = [x * x for x in numbers if x % 2 == 0]
        
        # Using map and filter
        squares_map = list(map(lambda x: x * x, filter(lambda x: x % 2 == 0, numbers)))
        
        return squares_comp, squares_map
        
    @staticmethod
    def dict_manipulation() -> Dict:
        """
        Demonstrates dictionary manipulation and defaultdict usage
        
        Returns:
            Dict: Processed dictionary
        """
        # Using defaultdict
        grades = [('Alice', 'A'), ('Bob', 'B'), ('Alice', 'A+'), ('Charlie', 'C')]
        student_grades = defaultdict(list)
        
        for student, grade in grades:
            student_grades[student].append(grade)
            
        # Convert to regular dict for return
        return dict(student_grades)
        
    @staticmethod
    def generator_example(n: int):
        """
        Demonstrates generator function usage
        
        Args:
            n (int): Number of Fibonacci numbers to generate
        
        Yields:
            int: Next Fibonacci number
        """
        a, b = 0, 1
        for _ in range(n):
            yield a
            a, b = b, a + b
            
    @classmethod
    def json_handling(cls) -> str:
        """
        Demonstrates JSON handling in Python
        
        Returns:
            str: JSON string
        """
        data = {
            'name': 'Python Interview',
            'topics': ['generators', 'decorators', 'context managers'],
            'date': str(datetime.now())
        }
        
        return json.dumps(data, indent=2)
        
    @staticmethod
    def string_manipulation(s: str) -> Dict:
        """
        Demonstrates various string manipulations
        
        Args:
            s (str): Input string
            
        Returns:
            Dict: Various string operation results
        """
        return {
            'reversed': s[::-1],
            'uppercase': s.upper(),
            'word_count': len(s.split()),
            'char_frequency': Counter(s.lower())
        }
        
    def __str__(self) -> str:
        """String representation of the class"""
        return "Python Interview Solutions - Common Questions and Patterns"

# Test cases
if __name__ == "__main__":
    solutions = PythonInterviewSolutions()
    
    # Test list comprehension vs map
    comp, map_result = solutions.list_comprehension_vs_map()
    assert comp == map_result == [4, 16, 36, 64, 100]
    
    # Test dictionary manipulation
    grades_dict = solutions.dict_manipulation()
    assert grades_dict['Alice'] == ['A', 'A+']
    assert grades_dict['Bob'] == ['B']
    
    # Test generator
    fib_numbers = list(solutions.generator_example(5))
    assert fib_numbers == [0, 1, 1, 2, 3]
    
    # Test JSON handling
    json_str = solutions.json_handling()
    parsed = json.loads(json_str)
    assert 'name' in parsed
    assert 'topics' in parsed
    assert 'date' in parsed
    
    # Test string manipulation
    str_results = solutions.string_manipulation("Hello World")
    assert str_results['reversed'] == "dlroW olleH"
    assert str_results['uppercase'] == "HELLO WORLD"
    assert str_results['word_count'] == 2
    
    print("All Python interview solution tests passed!")