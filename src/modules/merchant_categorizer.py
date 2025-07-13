import os
import re
import json
from datetime import datetime
from pathlib import Path
from src.config.paths import MERCHANT_DB_PATH, MERCHANT_ALIASES_PATH, MANUAL_CATEGORIES_PATH, paths

class MerchantCategorizer:
    """
    A class for categorizing transactions based on merchant names with auto-learning capability.
    This supplements the existing pattern-based categorization with a more precise merchant-based approach.
    """
    
    def __init__(self, base_path=None):
        """Initialize the merchant categorizer with optional base path."""
        if base_path is None:
            base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.base_path = base_path
        self.merchant_db_path = MERCHANT_DB_PATH
        self.alias_db_path = MERCHANT_ALIASES_PATH
        
        # Create cached_data directory if it doesn't exist
        paths.ensure_cached_data_exists()
        
        # Load or create merchant database
        self.merchant_db = self.load_merchant_db()
        self.alias_db = self.load_alias_db()
        
        print(f"Loaded merchant database with {len(self.merchant_db)} merchants and {len(self.alias_db)} aliases")
        
    def load_merchant_db(self):
        """Load the merchant database from file."""
        try:
            if os.path.exists(self.merchant_db_path):
                with open(self.merchant_db_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading merchant database: {str(e)}")
            return {}

    def save_merchant_db(self):
        """Save the merchant database to file."""
        try:
            with open(self.merchant_db_path, 'w') as f:
                json.dump(self.merchant_db, f, indent=2)
        except Exception as e:
            print(f"Error saving merchant database: {str(e)}")

    def load_alias_db(self):
        """Load the alias database from file."""
        try:
            if os.path.exists(self.alias_db_path):
                with open(self.alias_db_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading alias database: {str(e)}")
            return {}

    def save_alias_db(self):
        """Save the alias database to file."""
        try:
            with open(self.alias_db_path, 'w') as f:
                json.dump(self.alias_db, f, indent=2)
        except Exception as e:
            print(f"Error saving alias database: {str(e)}")

    def add_merchant(self, merchant_name: str, category: str) -> bool:
        """Add or update a merchant's category."""
        try:
            self.merchant_db[merchant_name.lower()] = category
            self.save_merchant_db()
            return True
        except Exception as e:
            print(f"Error adding merchant: {str(e)}")
            return False

    def delete_merchant(self, merchant_name: str) -> bool:
        """Delete a merchant from the database."""
        try:
            merchant_name = merchant_name.lower()
            if merchant_name in self.merchant_db:
                del self.merchant_db[merchant_name]
                self.save_merchant_db()
                
                # Also remove any aliases for this merchant
                self.alias_db = {alias: merchant for alias, merchant in self.alias_db.items() 
                               if merchant != merchant_name}
                self.save_alias_db()
                return True
            return False
        except Exception as e:
            print(f"Error deleting merchant: {str(e)}")
            return False

    def add_alias(self, alias: str, merchant: str) -> bool:
        """Add an alias for a merchant."""
        try:
            self.alias_db[alias.lower()] = merchant.lower()
            self.save_alias_db()
            return True
        except Exception as e:
            print(f"Error adding alias: {str(e)}")
            return False

    def get_all_merchants(self) -> dict:
        """Get all merchants and their categories."""
        return self.merchant_db

    def get_all_aliases(self) -> dict:
        """Get all aliases and their merchants."""
        return self.alias_db

    def search_merchants(self, search_term: str) -> dict:
        """Search merchants by name."""
        search_term = search_term.lower()
        return {name: category for name, category in self.merchant_db.items() 
                if search_term in name}

    def auto_learn(self, transaction_details: list, category: str) -> None:
        """Auto-learn merchant-category mapping from transaction details."""
        # Extract potential merchant name from transaction details
        merchant_name = self.extract_merchant_name(transaction_details)
        if merchant_name and merchant_name.lower() not in self.merchant_db:
            self.add_merchant(merchant_name, category)

    def extract_merchant_name(self, transaction_details: list) -> str:
        """Extract merchant name from transaction details."""
        if not transaction_details:
            return None
        
        # Join all details into a single string
        details_str = ' '.join(transaction_details)
        
        # Remove common prefixes/suffixes and special characters
        cleaned_str = re.sub(r'[^a-zA-Z0-9\s]', ' ', details_str)
        cleaned_str = re.sub(r'\s+', ' ', cleaned_str).strip()
        
        # Return the cleaned string as the merchant name
        return cleaned_str if cleaned_str else None

    def categorize_transaction(self, transaction_details, default_category="uncategorized"):
        """
        Categorize a transaction based on merchant matching
        
        Args:
            transaction_details: The transaction details (string or list)
            default_category: Category to use if no match is found
            
        Returns:
            tuple: (category, merchant_name)
        """
        # Convert transaction details to string and lowercase
        if isinstance(transaction_details, list):
            details = ' '.join(transaction_details).lower()
        else:
            details = str(transaction_details).lower()
        
        # Clean up the details
        details = re.sub(r'[^a-z\s&]', '', details)
        
        # Check for exact merchant matches
        for merchant, category in self.merchant_db.items():
            merchant_lower = merchant.lower()
            if merchant_lower in details or details in merchant_lower:
                return category, merchant
        
        # Check for alias matches
        for alias, merchant in self.alias_db.items():
            alias_lower = alias.lower()
            if (alias_lower in details or details in alias_lower) and merchant in self.merchant_db:
                return self.merchant_db[merchant], merchant
        
        # Try word-by-word matching for merchants
        details_words = set(details.split())
        for merchant, category in self.merchant_db.items():
            merchant_words = set(merchant.lower().split())
            # If all words in the merchant name are found in the details
            if merchant_words.issubset(details_words) or details_words.issubset(merchant_words):
                return category, merchant
        
        # Extract merchant name for unmatched transactions
        merchant = self.extract_merchant(details)
        return default_category, merchant

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
    
    def save_db(self):
        """Save the merchant database to disk"""
        # Update metadata
        self.merchant_db["metadata"]["last_updated"] = datetime.now().isoformat()
        self.merchant_db["metadata"]["merchant_count"] = len(self.merchant_db)
        self.merchant_db["metadata"]["alias_count"] = len(self.alias_db)
        
        # Write to disk
        self.save_merchant_db()
            
        # Update alias database with new patterns if needed
        if os.path.exists(self.alias_db_path):
            try:
                with open(self.alias_db_path, 'r') as f:
                    alias_db = json.load(f)
                
                # For each merchant in our database
                for merchant, category in self.merchant_db.items():
                    # Skip if category doesn't exist in alias database
                    if category not in alias_db.get('categories', {}):
                        continue
                        
                    # Check if we need to add a pattern for this merchant
                    patterns = alias_db['categories'][category]['patterns']
                    merchant_terms = merchant.split()
                    pattern_exists = any(
                        all(term in ' '.join(p.get('terms', [])).lower() for term in merchant_terms)
                        for p in patterns
                    )
                    
                    if not pattern_exists:
                        # Add new pattern
                        new_pattern = {
                            "terms": merchant_terms,
                            "dateAdded": datetime.now().isoformat(),
                            "lastUpdated": datetime.now().isoformat(),
                            "matchCount": 0
                        }
                        patterns.append(new_pattern)
                
                # Save updated alias database
                self.save_alias_db()
                    
            except Exception as e:
                print(f"Error updating alias database: {str(e)}")
    
    def search_merchants(self, search_term):
        """Search for merchants matching the search term"""
        search_term = search_term.lower()
        results = {}
        
        # Search in merchants
        for merchant, category in self.merchant_db.items():
            if search_term in merchant:
                results[merchant] = category
        
        # Search in aliases
        for alias, canonical in self.alias_db.items():
            if search_term in alias:
                results[f"{alias} -> {canonical}"] = self.merchant_db[canonical]
                
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
        if category == "uncategorized":
            return False
            
        # Extract merchant
        merchant = self.extract_merchant(transaction_details)
        if merchant == "Unknown":
            return False
            
        # Only add if this is a new merchant
        if merchant not in self.merchant_db and merchant not in self.alias_db:
            self.add_merchant(merchant, category)
            return True
            
        return False

class ManualTransactionCategorizer:
    """
    A class for handling manual category assignments for specific transactions.
    """
    
    def __init__(self, base_path=None):
        """Initialize the manual transaction categorizer with optional base path."""
        if base_path is None:
            base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.base_path = base_path
        self.manual_categories_path = MANUAL_CATEGORIES_PATH
        
        # Create cached_data directory if it doesn't exist
        paths.ensure_cached_data_exists()
        
        # Load or create manual categories database
        self.manual_categories = self.load_manual_categories()
        
    def load_manual_categories(self):
        """Load the manual categories database from file."""
        try:
            if os.path.exists(self.manual_categories_path):
                with open(self.manual_categories_path, 'r') as f:
                    data = json.load(f)
                    # Convert old format to new format if needed
                    if isinstance(data, dict):
                        new_data = []
                        for transaction_id, category in data.items():
                            date_time, amount = transaction_id.split('_')
                            new_data.append({
                                'datetime': date_time,
                                'amount': float(amount),
                                'category': category,
                                'index': None  # Will be set when used
                            })
                        return new_data
                    return data
            return []
        except Exception as e:
            print(f"Error loading manual categories database: {str(e)}")
            return []

    def save_manual_categories(self):
        """Save the manual categories database to file."""
        try:
            with open(self.manual_categories_path, 'w') as f:
                json.dump(self.manual_categories, f, indent=2)
        except Exception as e:
            print(f"Error saving manual categories database: {str(e)}")

    def add_manual_category(self, transaction_id: str, category: str) -> bool:
        """Add a manual category for a specific transaction."""
        try:
            # Split the composite key
            date_time, amount = transaction_id.split('_')
            amount = float(amount)
            
            # Remove any existing entry for this transaction
            self.manual_categories = [entry for entry in self.manual_categories 
                                    if not (entry['datetime'] == date_time and entry['amount'] == amount)]
            
            # Add new entry
            self.manual_categories.append({
                'datetime': date_time,
                'amount': amount,
                'category': category,
                'index': None  # Will be set when used
            })
            
            self.save_manual_categories()
            return True
        except Exception as e:
            print(f"Error adding manual category: {str(e)}")
            return False

    def get_manual_category(self, transaction_id: str) -> str:
        """Get the manual category for a specific transaction if it exists."""
        try:
            date_time, amount = transaction_id.split('_')
            amount = float(amount)
            
            for entry in self.manual_categories:
                if entry['datetime'] == date_time and entry['amount'] == amount:
                    return entry['category']
            return None
        except Exception as e:
            print(f"Error getting manual category: {str(e)}")
            return None

    def remove_manual_category(self, transaction_id: str) -> bool:
        """Remove a manual category for a specific transaction."""
        try:
            date_time, amount = transaction_id.split('_')
            amount = float(amount)
            
            # Remove matching entry
            self.manual_categories = [entry for entry in self.manual_categories 
                                    if not (entry['datetime'] == date_time and entry['amount'] == amount)]
            
            self.save_manual_categories()
            return True
        except Exception as e:
            print(f"Error removing manual category: {str(e)}")
            return False 