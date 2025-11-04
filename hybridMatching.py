import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher
import unicodedata
import json

@dataclass
class ProductAttributes:
    """Extracted product attributes"""
    original_title: str
    cleaned_title: str
    brand: Optional[str]
    core_product: str
    size: Optional[str]
    color: Optional[str]
    quantity: Optional[str]
    category: str

@dataclass
class MatchResult:
    """Match result with confidence score"""
    is_match: bool
    confidence_score: float
    needs_review: bool
    match_details: Dict[str, any]

class ProductMatcher:
    """
    Advanced product matching algorithm for Amazon and SERP products.
    Handles exact matching with support for variants, multi-packs, and attribute variations.
    """
    
    def __init__(self):
        # Noise words to remove during preprocessing
        self.noise_words = {
            'sale', 'deal', 'offer', 'discount', 'authentic', 'genuine', 
            'original', 'new', 'latest', 'model', 'fast', 'shipping',
            'free', 'delivery', 'prime', 'bestseller', 'top', 'rated',
            'official', 'certified', 'authorized', 'shop', 'store'
        }
        
        # Size/capacity patterns
        self.size_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(ml|milliliter|millilitre|l|liter|litre|oz|ounce|fl\.?\s?oz|gallon|gal)\b',
            r'\b(\d+(?:\.\d+)?)\s*(mg|milligram|g|gram|kg|kilogram|lb|pound)\b',
            r'\b(\d+(?:\.\d+)?)\s*(mm|cm|centimeter|m|meter|inch|in|ft|foot|feet)\b',
            r'\b(x-small|xs|small|s|medium|m|large|l|x-large|xl|xx-large|xxl|2xl|3xl)\b'
        ]
        
        # Quantity patterns
        self.quantity_patterns = [
            r'\b(?:pack\s+of|count\s+of|\bx)\s+(\d+)\b',
            r'\b(\d+)\s*(?:pack|pk|count|ct|piece|pcs|pcs\.|pc)\b',
            r'\b(single|double|twin|triple|quad)\s+pack\b',
            r'\b(\d+)[-\s]pack\b'
        ]
        
        # Color patterns (common colors)
        self.color_keywords = {
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple',
            'orange', 'brown', 'grey', 'gray', 'silver', 'gold', 'beige', 'navy',
            'crimson', 'scarlet', 'azure', 'cyan', 'lime', 'olive', 'maroon',
            'violet', 'indigo', 'turquoise', 'tan', 'ivory', 'pearl', 'rose',
            'coral', 'salmon', 'peach', 'mint', 'lavender', 'mauve', 'cream'
        }
        
        # Unit conversions for normalization
        self.unit_conversions = {
            # Volume (base: ml)
            'ml': 1, 'milliliter': 1, 'millilitre': 1,
            'l': 1000, 'liter': 1000, 'litre': 1000,
            'floz': 29.5735, 'fl oz': 29.5735, 'fl.oz': 29.5735,
            'gallon': 3785.41, 'gal': 3785.41,
            # Weight (base: grams)
            'mg': 0.001, 'milligram': 0.001,
            'g': 1, 'gram': 1,
            'kg': 1000, 'kilogram': 1000,
            'oz': 28.3495, 'ounce': 28.3495,  # Weight ounces
            'lb': 453.592, 'pound': 453.592,
            # Length (base: mm)
            'mm': 1, 'millimeter': 1,
            'cm': 10, 'centimeter': 10,
            'm': 1000, 'meter': 1000,
            'inch': 25.4, 'in': 25.4,
            'ft': 304.8, 'foot': 304.8, 'feet': 304.8
        }
        
        # Quantity word to number mapping
        self.quantity_words = {
            'single': '1', 'double': '2', 'twin': '2',
            'triple': '3', 'quad': '4'
        }
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters (e.g., M·A·C -> MAC)"""
        # Remove accents and special unicode
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ASCII', 'ignore').decode('ASCII')
        return text
    
    def preprocess_title(self, title: str) -> str:
        """Clean and normalize product title"""
        # Convert to lowercase
        title = title.lower()
        
        # Normalize unicode
        title = self.normalize_unicode(title)
        
        # Remove special characters but keep important ones
        title = re.sub(r'[^\w\s\-\.\,]', ' ', title)
        
        # Normalize spaces
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove noise words
        words = title.split()
        filtered_words = [w for w in words if w not in self.noise_words]
        title = ' '.join(filtered_words)
        
        return title
    
    def extract_brand(self, title: str) -> Optional[str]:
        """Extract brand name (assuming it's typically at the start)"""
        # Brand is usually the first 1-3 words
        words = title.split()
        if len(words) >= 1:
            # Check if first word looks like a brand (capitalized or known pattern)
            potential_brand = words[0]
            # Could be extended with a brand database lookup
            return potential_brand
        return None
    
    def extract_size(self, title: str) -> Optional[Tuple[str, float, str]]:
        """Extract size/capacity with normalization"""
        for pattern in self.size_patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                # Check if this is a text-based size (small, medium, large, etc.)
                size_text = match.group(0).lower()
                if any(size_word in size_text for size_word in ['x-small', 'xs', 'small', 'medium', 'large', 'xl', 'xxl', '2xl', '3xl']):
                    # Return the text size as-is without numeric conversion
                    return (match.group(0), 0.0, size_text.strip())
                
                # Try to convert to float, skip if it fails
                try:
                    value = float(match.group(1))
                except (ValueError, AttributeError):
                    # If can't convert to float, it's likely a text size descriptor
                    return (match.group(0), 0.0, match.group(0).lower().strip())
                
                unit = match.group(2).lower()
                
                # Normalize unit
                normalized_unit = unit.replace('.', '').replace(' ', '')
                base_value = value
                
                # Convert to base unit if possible
                if normalized_unit in self.unit_conversions:
                    base_value = value * self.unit_conversions[normalized_unit]
                
                return (match.group(0), base_value, normalized_unit)
        return None
    
    def extract_quantity(self, title: str) -> Optional[Tuple[str, int]]:
        """Extract quantity/pack information"""
        for pattern in self.quantity_patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                quantity_str = match.group(1)
                
                # Convert word to number if needed
                if quantity_str.lower() in self.quantity_words:
                    quantity_str = self.quantity_words[quantity_str.lower()]
                
                try:
                    quantity = int(quantity_str)
                    return (match.group(0), quantity)
                except ValueError:
                    continue
        
        # Default to single item if no quantity found
        return None
    
    def extract_color(self, title: str) -> Optional[str]:
        """Extract color from title"""
        words = title.lower().split()
        for word in words:
            if word in self.color_keywords:
                return word
        return None
    
    def extract_attributes(self, title: str, category: str) -> ProductAttributes:
        """Extract all relevant attributes from product title"""
        cleaned = self.preprocess_title(title)
        
        # Extract attributes
        brand = self.extract_brand(cleaned)
        size_info = self.extract_size(cleaned)
        quantity_info = self.extract_quantity(cleaned)
        color = self.extract_color(cleaned)
        
        # Create core product name by removing extracted attributes
        core_product = cleaned
        
        if size_info:
            core_product = core_product.replace(size_info[0], '')
        if quantity_info:
            core_product = core_product.replace(quantity_info[0], '')
        
        # Clean up core product
        core_product = re.sub(r'\s+', ' ', core_product).strip()
        
        return ProductAttributes(
            original_title=title,
            cleaned_title=cleaned,
            brand=brand,
            core_product=core_product,
            size=size_info[0] if size_info else None,
            color=color,
            quantity=str(quantity_info[1]) if quantity_info else '1',
            category=category
        )
    
    def calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using sequence matching"""
        return SequenceMatcher(None, str1, str2).ratio()
    
    def compare_sizes(self, size1: Optional[Tuple], size2: Optional[Tuple]) -> float:
        """Compare size attributes with normalization and 5% tolerance"""
        if size1 is None and size2 is None:
            return 1.0  # Both don't have size
        if size1 is None or size2 is None:
            return 0.5  # One has size, other doesn't
        
        # Extract values and units
        _, value1, unit1 = size1
        _, value2, unit2 = size2
        
        # Handle text-based sizes (small, medium, large, etc.)
        if value1 == 0.0 and value2 == 0.0:
            # Both are text sizes, compare directly
            return 1.0 if unit1 == unit2 else 0.0
        elif value1 == 0.0 or value2 == 0.0:
            # One is text, one is numeric - treat as different
            return 0.0
        
        # Determine measurement types (volume or weight)
        volume_units = {'ml', 'milliliter', 'millilitre', 'l', 'liter', 'litre', 
                    'fl oz', 'floz', 'fl.oz', 'gallon', 'gal'}
        weight_units = {'mg', 'milligram', 'g', 'gram', 'kg', 'kilogram', 'lb', 'pound', 'oz', 'ounce'}
        
        # Normalize unit strings for comparison
        unit1_normalized = unit1.replace('.', '').replace(' ', '').lower()
        unit2_normalized = unit2.replace('.', '').replace(' ', '').lower()
        
        # Check if units are in the same category
        unit1_is_volume = unit1_normalized in volume_units
        unit2_is_volume = unit2_normalized in volume_units
        unit1_is_weight = unit1_normalized in weight_units
        unit2_is_weight = unit2_normalized in weight_units
        
        # If different measurement types (volume vs weight), they don't match
        if (unit1_is_volume and unit2_is_weight) or (unit1_is_weight and unit2_is_volume):
            return 0.0
        
        # Both are same type - convert to common base unit and compare
        # value1 and value2 are already in base units from extract_size
        
        # Calculate percentage difference
        larger_value = max(value1, value2)
        difference = abs(value1 - value2)
        
        if larger_value == 0:
            return 0.0
        
        percentage_diff = (difference / larger_value) * 100
        
        # Allow 5% tolerance
        if percentage_diff <= 5.0:
            return 1.0
        else:
            return 0.0  # Different sizes = variant
    
    def compare_quantities(self, qty1: str, qty2: str) -> float:
        """Compare quantity attributes"""
        if qty1 == qty2:
            return 1.0
        return 0.0
    
    def compare_colors(self, color1: Optional[str], color2: Optional[str]) -> float:
        """Compare color attributes"""
        if color1 is None and color2 is None:
            return 1.0
        if color1 is None or color2 is None:
            return 0.7  # Partial penalty if one missing
        if color1 == color2:
            return 1.0
        return 0.0  # Different colors = variant
    
    def match_products(self, 
                      amazon_title: str, 
                      serp_title: str, 
                      category: str) -> MatchResult:
        """
        Main matching function that compares two product titles.
        
        Args:
            amazon_title: Product title from Amazon
            serp_title: Product title from SERP
            category: Product category
            
        Returns:
            MatchResult with confidence score and match decision
        """
        # Extract attributes from both titles
        amazon_attrs = self.extract_attributes(amazon_title, category)
        serp_attrs = self.extract_attributes(serp_title, category)
        
        # Initialize match details
        match_details = {
            'amazon_attributes': amazon_attrs.__dict__,
            'serp_attributes': serp_attrs.__dict__,
            'scores': {}
        }
        
        # 1. Brand matching (CRITICAL - must match)
        brand_score = 0.0
        if amazon_attrs.brand and serp_attrs.brand:
            brand_score = self.calculate_string_similarity(
                amazon_attrs.brand, 
                serp_attrs.brand
            )
            match_details['scores']['brand'] = brand_score
            
            # Brand must have high similarity (>0.8) to continue
            if brand_score < 0.8:
                return MatchResult(
                    is_match=False,
                    confidence_score=brand_score * 100,
                    needs_review=False,
                    match_details=match_details
                )
        else:
            # If brand missing in either, it's a no match
            match_details['scores']['brand'] = 0.0
            return MatchResult(
                is_match=False,
                confidence_score=0.0,
                needs_review=False,
                match_details=match_details
            )
        
        # 2. Core product similarity (product name without attributes)
        core_similarity = self.calculate_string_similarity(
            amazon_attrs.core_product,
            serp_attrs.core_product
        )
        match_details['scores']['core_product'] = core_similarity
        
        # 3. Size comparison (must be exact match)
        size1 = self.extract_size(amazon_attrs.cleaned_title)
        size2 = self.extract_size(serp_attrs.cleaned_title)
        size_score = self.compare_sizes(size1, size2)
        match_details['scores']['size'] = size_score
        
        # 4. Quantity comparison (must be exact match)
        quantity_score = self.compare_quantities(
            amazon_attrs.quantity,
            serp_attrs.quantity
        )
        match_details['scores']['quantity'] = quantity_score
        
        # 5. Color comparison (must be exact match if present)
        color_score = self.compare_colors(
            amazon_attrs.color,
            serp_attrs.color
        )
        match_details['scores']['color'] = color_score
        
        # 6. Calculate weighted final score
        # Brand and core product are most important
        # Size, quantity, color must be exact (1.0) or it's a variant
        
        weights = {
            'brand': 0.25,
            'core_product': 0.40,
            'size': 0.15,
            'quantity': 0.10,
            'color': 0.10
        }
        
        final_score = (
            brand_score * weights['brand'] +
            core_similarity * weights['core_product'] +
            size_score * weights['size'] +
            quantity_score * weights['quantity'] +
            color_score * weights['color']
        )
        
        # Apply hard constraints: if size, quantity, or color don't match exactly, cap score
        if size_score < 1.0 or quantity_score < 1.0 or color_score < 1.0:
            # These are variants, not exact matches
            final_score = min(final_score, 0.75)
        
        # Convert to percentage
        confidence_percentage = final_score * 100
        match_details['final_confidence'] = confidence_percentage
        
        # Determine match status based on thresholds
        is_match = confidence_percentage >= 90
        needs_review = 80 <= confidence_percentage < 90
        
        # print(is_match, needs_review)

        if not is_match:
            print("Amazon Title:", amazon_title)
            print("SERP Title:", serp_title)
            # print("Match Details:", match_details)
            print(match_details['amazon_attributes']['brand'], "<->", match_details['serp_attributes']['brand'])
            print(match_details['amazon_attributes']['size'], "<->", match_details['serp_attributes']['size'])
            print(match_details['amazon_attributes']['quantity'], "<->", match_details['serp_attributes']['quantity'])
            print(match_details['amazon_attributes']['color'], "<->", match_details['serp_attributes']['color'])

            print("\n\n")

        return MatchResult(
            is_match=is_match,
            confidence_score=confidence_percentage,
            needs_review=needs_review,
            match_details=match_details
        )
    
    def match_against_multiple(self,
                               amazon_title: str,
                               serp_titles: List[str],
                               category: str) -> List[Tuple[str, MatchResult]]:
        """
        Match one Amazon product against multiple SERP products.
        
        Args:
            amazon_title: Amazon product title
            serp_titles: List of SERP product titles (max 20)
            category: Product category
            
        Returns:
            List of tuples (serp_title, MatchResult) sorted by confidence
        """
        results = []
        
        for serp_title in serp_titles:
            match_result = self.match_products(amazon_title, serp_title, category)
            results.append((serp_title, match_result))
        
        # Sort by confidence score (descending)
        results.sort(key=lambda x: x[1].confidence_score, reverse=True)
        
        return results


# Example usage
if __name__ == "__main__":
    matcher = ProductMatcher()
    
    with open("input.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Test cases
    # test_cases = [
    #     {
    #         'amazon': 'CeraVe Moisturizing Cream 16 oz',
    #         'serp': [
    #             'CeraVe Moisturizing Cream 16 oz Daily Face and Body Moisturizer',
    #             'CeraVe Moisturizing Cream 8 oz',
    #             'CeraVe Facial Moisturizing Lotion 16 oz',
    #             'Cerave Moisturizing Cream 16oz [SALE]'
    #         ],
    #         'category': 'Beauty'
    #     },
    #     {
    #         'amazon': 'Apple iPhone 15 Pro 128GB Blue',
    #         'serp': [
    #             'Apple iPhone 15 Pro 128GB Blue Titanium',
    #             'Apple iPhone 15 Pro 256GB Blue',
    #             'iPhone 15 Pro 128GB Blue',
    #             'Apple iPhone 15 Pro Max 128GB Blue'
    #         ],
    #         'category': 'Electronics'
    #     },
    #     {
    #         'amazon': 'Nivea Creme 30ml Pack of 3',
    #         'serp': [
    #             'Nivea Creme 30ml 3-Pack',
    #             'Nivea Creme 30ml Single',
    #             'Nivea Creme 60ml Pack of 3',
    #             'Authentic Nivea Creme 30ml x 3'
    #         ],
    #         'category': 'Beauty'
    #     }
    # ]
    
    # Run tests
    for i, test in enumerate(data, 1):
        # print(f"\n{'='*80}")
        # print(f"Test Case {i}: {test['category']}")
        # print(f"Amazon Product: {test['title']}")
        # print(f"{'='*80}")
        
        results = matcher.match_against_multiple(
            test['title'],
            test['seoTitles'],
            test['category']
        )
        
        for serp_title, result in results:
            status = "✓ MATCH" if result.is_match else "⚠ REVIEW" if result.needs_review else "✗ NO MATCH"
            # print(f"\n{status} [{result.confidence_score:.2f}%] {serp_title}")
            
            # Show key matching details
            scores = result.match_details['scores']
            # print(f"  Brand: {scores.get('brand', 0):.2f} | Core: {scores.get('core_product', 0):.2f} | "
                #   f"Size: {scores.get('size', 0):.2f} | Qty: {scores.get('quantity', 0):.2f} | "
                #   f"Color: {scores.get('color', 0):.2f}")

    results_list = []

    for i, test in enumerate(data, 1):
        test_case_data = {
            "test_case": i,
            "category": test["category"],
            "amazon_product": test["title"],
            "matches": []
        }

        results = matcher.match_against_multiple(
            test["title"],
            test["seoTitles"],
            test["category"]
        )

        for serp_title, result in results:
            # Determine status
            if result.is_match:
                status = "MATCH"
            elif result.needs_review:
                status = "REVIEW"
            else:
                status = "NO MATCH"

            scores = result.match_details["scores"]
            match_entry = {
                "serp_title": serp_title,
                "status": status,
                "confidence_score": round(result.confidence_score, 2),
                "scores": {
                    "brand": round(scores.get("brand", 0), 2),
                    "core_product": round(scores.get("core_product", 0), 2),
                    "size": round(scores.get("size", 0), 2),
                    "quantity": round(scores.get("quantity", 0), 2),
                    "color": round(scores.get("color", 0), 2),
                }
            }

            test_case_data["matches"].append(match_entry)

        results_list.append(test_case_data)

    # Save to JSON file
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=4, ensure_ascii=False)


# matcher = ProductMatcher()

# # Single comparison
# result = matcher.match_products(
#     amazon_title="CeraVe Cream 16oz",
#     serp_title="CeraVe Cream 16 oz Daily Moisturizer",
#     category="Beauty"
# )

# # Multiple comparisons (1 vs 20)
# results = matcher.match_against_multiple(
#     amazon_title="KONG Classic Dog Toy, Red, Large, Durable Natural Rubber Chew Toy",
#     serp_titles=[
#       "KONG Classic Dog Toy – Large Red Durable Rubber",
#       "KONG Red Dog Toy – Tough Chew Toy for Large Dogs",
#       "KONG Classic – Large Size Rubber Toy | SALE",
#       "KONG Durable Dog Chew Toy – Free Shipping Deal",
#       "KONG Classic Large Dog Toy – Long-Lasting Rubber",
#       "KONG Dog Toy – Large Red Chew | Pet Supplies",
#       "KONG Classic – Durable Toy for Active Dogs"
#     ],
#     category="Pet Supplies"
# )