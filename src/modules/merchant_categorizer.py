import os
import re
import json
from datetime import datetime

class MerchantCategorizer:
    """
    A class for categorizing transactions based on merchant names with auto-learning capability.
    This supplements the existing pattern-based categorization with a more precise merchant-based approach.
    """
    
    def __init__(self, base_path=None):
        """Initialize the merchant categorizer"""
        self.base_path = base_path
        # Set up the merchant database path
        if base_path:
            self.db_path = os.path.join(base_path, "cached_data", "merchant_db.json")
        else:
            self.db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cached_data", "merchant_db.json")
        
        # Load or create the merchant database
        self.merchant_db = self._load_or_create_db()
        
    def _load_or_create_db(self):
        """Load existing merchant database or create a new one if it doesn't exist"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        if not os.path.exists(self.db_path):
            # Create a new merchant database
            merchant_db = {
                "merchants": {},  # Will store merchant_name: category
                "aliases": {},    # Will store alias: canonical_name
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "version": "1.0",
                    "merchant_count": 0,
                    "alias_count": 0
                }
            }
            # Save the initial db
            with open(self.db_path, 'w') as f:
                json.dump(merchant_db, f, indent=2)
            print(f"Created new merchant database at {self.db_path}")
        else:
            # Load existing database
            with open(self.db_path, 'r') as f:
                merchant_db = json.load(f)
            print(f"Loaded merchant database with {len(merchant_db['merchants'])} merchants and {len(merchant_db['aliases'])} aliases")
        
        return merchant_db
    
    def save_db(self):
        """Save the merchant database to disk"""
        # Update metadata
        self.merchant_db["metadata"]["last_updated"] = datetime.now().isoformat()
        self.merchant_db["metadata"]["merchant_count"] = len(self.merchant_db["merchants"])
        self.merchant_db["metadata"]["alias_count"] = len(self.merchant_db["aliases"])
        
        # Write to disk
        with open(self.db_path, 'w') as f:
            json.dump(self.merchant_db, f, indent=2)
    
    def extract_merchant(self, transaction_details):
        """Extract the likely merchant name from transaction details"""
        # Join the details into a single string if it's a list
        if isinstance(transaction_details, list):
            details = ' '.join(transaction_details)
        else:
            details = transaction_details
            
        # Convert to lowercase for consistency
        details = details.lower()
            
        # Remove common transaction prefixes/suffixes
        details = re.sub(r'(purchase|payment|pos|point of sale|debit|credit|#\d+|wire transfer|etransfer|transfer)', '', details, flags=re.IGNORECASE)
        
        # Remove common financial terms
        details = re.sub(r'\b(fee|interest|charge|deposit|withdrawal|service|closing|balance|opening)\b', '', details, flags=re.IGNORECASE)
        
        # Remove locations (usually city/state/country format)
        details = re.sub(r'\b[a-z]+ [a-z]{2}\b', '', details, flags=re.IGNORECASE)
        details = re.sub(r'\b(on|ca|qc|bc|ab|sk|mb|nb|ns|nl|pe|nt|yt|nu)\b', '', details, flags=re.IGNORECASE)
        
        # Remove dates and times
        details = re.sub(r'\d{1,2}/\d{1,2}(/\d{2,4})?', '', details)
        details = re.sub(r'\d{1,2}:\d{2}', '', details)
        
        # Remove common numbers and special characters
        details = re.sub(r'[#*.,0-9]', '', details)
        
        # Clean up extra spaces
        details = re.sub(r'\s+', ' ', details).strip()
        
        # If details are now empty, return Unknown
        if not details:
            return "Unknown"
        
        # Split by spaces and get words
        words = [w for w in details.split() if len(w) > 1]
        if not words:
            return "Unknown"
        
        # Try to return just the merchant name (typically first 1-3 words)
        if len(words) <= 3:
            return ' '.join(words).strip()
        else:
            # For longer strings, prioritize finding known merchant words
            potential_merchant_words = ["restaurant", "cafe", "shop", "store", "market", "supermarket", 
                                       "walmart", "costco", "amazon", "starbucks", "mcdonalds", "uber", 
                                       "lyft", "pizza", "doordash", "tim hortons", "subway"]
            
            for merchant_word in potential_merchant_words:
                if merchant_word in details:
                    # Find the word and include a window around it
                    word_idx = details.find(merchant_word)
                    start_idx = max(0, details.rfind(' ', 0, word_idx))
                    end_idx = details.find(' ', word_idx + len(merchant_word))
                    if end_idx == -1:
                        end_idx = len(details)
                    return details[start_idx:end_idx].strip()
            
            # Default to first 3 words if no merchant word found
            return ' '.join(words[:3]).strip()
    
    def categorize_transaction(self, transaction_details, default_category="uncharacterized"):
        """
        Categorize a transaction based on merchant matching
        
        Args:
            transaction_details: The transaction details (string or list)
            default_category: Category to use if no match is found
            
        Returns:
            tuple: (category, merchant_name)
        """
        # Extract merchant
        merchant = self.extract_merchant(transaction_details)
        
        # Check if we have this exact merchant
        if merchant in self.merchant_db["merchants"]:
            return self.merchant_db["merchants"][merchant], merchant
        
        # Check if this is an alias
        if merchant in self.merchant_db["aliases"]:
            canonical_name = self.merchant_db["aliases"][merchant]
            return self.merchant_db["merchants"][canonical_name], canonical_name
        
        # No match found
        return default_category, merchant
    
    def add_merchant(self, merchant_name, category):
        """Add or update a merchant in the database"""
        self.merchant_db["merchants"][merchant_name] = category
        self.save_db()
        return True
    
    def add_alias(self, alias, canonical_name):
        """Add an alias for an existing merchant"""
        # Ensure the canonical name exists
        if canonical_name not in self.merchant_db["merchants"]:
            return False
        
        # Add the alias
        self.merchant_db["aliases"][alias] = canonical_name
        self.save_db()
        return True
    
    def get_all_merchants(self):
        """Get all merchants in the database"""
        return self.merchant_db["merchants"]
    
    def get_all_aliases(self):
        """Get all aliases in the database"""
        return self.merchant_db["aliases"]
    
    def search_merchants(self, search_term):
        """Search for merchants matching the search term"""
        search_term = search_term.lower()
        results = {}
        
        # Search in merchants
        for merchant, category in self.merchant_db["merchants"].items():
            if search_term in merchant.lower():
                results[merchant] = category
        
        # Search in aliases
        for alias, canonical in self.merchant_db["aliases"].items():
            if search_term in alias.lower():
                results[f"{alias} -> {canonical}"] = self.merchant_db["merchants"][canonical]
                
        return results
    
    def auto_learn(self, transaction_details, category):
        """
        Automatically learn a merchant-category mapping from a categorized transaction
        
        Args:
            transaction_details: The transaction details (string or list)
            category: The assigned category
            
        Returns:
            bool: True if a new merchant mapping was added, False otherwise
        """
        if category == "uncharacterized":
            return False
            
        # Extract merchant
        merchant = self.extract_merchant(transaction_details)
        if merchant == "Unknown":
            return False
            
        # Only add if this is a new merchant
        if merchant not in self.merchant_db["merchants"] and merchant not in self.merchant_db["aliases"]:
            self.merchant_db["merchants"][merchant] = category
            self.save_db()
            return True
            
        return False 