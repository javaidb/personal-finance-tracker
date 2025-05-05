import json
import os
from pathlib import Path
from typing import Dict, List, Any
from src.modules.merchant_categorizer import MerchantCategorizer
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MerchantService:
    """Service class for handling merchant-related operations."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.merchant_categorizer = MerchantCategorizer(base_path=base_path)
        self.databank_path = os.path.join(base_path, 'cached_data', 'databank.json')
        self.review_path = os.path.join(base_path, 'cached_data', 'uncharacterized_merchants.json')
        self.load_databank()

    def load_databank(self):
        """Load or create databank.json"""
        try:
            if os.path.exists(self.databank_path):
                with open(self.databank_path, 'r') as f:
                    self.databank = json.load(f)
            else:
                # Create initial databank structure
                self.databank = {
                    "categories": {
                        "Groceries": {"totalMatches": 0, "patterns": []},
                        "Dining": {"totalMatches": 0, "patterns": []},
                        "Transport": {"totalMatches": 0, "patterns": []},
                        "Shopping": {"totalMatches": 0, "patterns": []},
                        "Bills": {"totalMatches": 0, "patterns": []},
                        "Entertainment": {"totalMatches": 0, "patterns": []},
                        "Activities": {"totalMatches": 0, "patterns": []},
                        "Online": {"totalMatches": 0, "patterns": []},
                        "Income": {"totalMatches": 0, "patterns": []},
                        "Rent": {"totalMatches": 0, "patterns": []},
                        "Investment": {"totalMatches": 0, "patterns": []}
                    }
                }
                self.save_databank()
        except Exception as e:
            logger.error(f"Error loading databank: {str(e)}", exc_info=True)
            self.databank = {"categories": {}}

    def save_databank(self):
        """Save databank to file"""
        try:
            with open(self.databank_path, 'w') as f:
                json.dump(self.databank, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving databank: {str(e)}", exc_info=True)

    def get_all_merchants(self) -> List[Dict[str, str]]:
        """Get all merchants with their categories."""
        try:
            merchants = self.merchant_categorizer.get_all_merchants()
            return [{"name": name, "category": category} for name, category in merchants.items()]
        except Exception as e:
            logger.error(f"Error getting all merchants: {str(e)}", exc_info=True)
            return []

    def search_merchants(self, search_term: str) -> List[Dict[str, str]]:
        """Search merchants by name."""
        try:
            merchants = self.merchant_categorizer.search_merchants(search_term)
            return [{"name": name, "category": category} for name, category in merchants.items()]
        except Exception as e:
            logger.error(f"Error searching merchants: {str(e)}", exc_info=True)
            return []

    def add_merchant(self, merchant_name: str, category: str) -> bool:
        """Add or update a merchant's category."""
        try:
            # First verify category exists in databank
            if category not in self.databank.get('categories', {}):
                return False

            # Add merchant to database
            success = self.merchant_categorizer.add_merchant(merchant_name, category)
            if success:
                # Add pattern to databank
                merchant_terms = merchant_name.lower().split()
                pattern = {
                    "terms": merchant_terms,
                    "dateAdded": datetime.now().isoformat(),
                    "lastUpdated": datetime.now().isoformat(),
                    "matchCount": 0
                }
                self.databank['categories'][category]['patterns'].append(pattern)
                self.save_databank()
            return success
        except Exception as e:
            logger.error(f"Error adding merchant: {str(e)}", exc_info=True)
            return False

    def add_alias(self, alias: str, merchant: str) -> bool:
        """Add an alias for a merchant."""
        try:
            return self.merchant_categorizer.add_alias(alias, merchant)
        except Exception as e:
            logger.error(f"Error adding alias: {str(e)}", exc_info=True)
            return False

    def get_categories(self) -> List[str]:
        """Get all available categories."""
        try:
            return list(self.databank.get('categories', {}).keys())
        except Exception as e:
            logger.error(f"Error loading categories: {str(e)}", exc_info=True)
            return ["Groceries", "Dining", "Transport", "Shopping", "Bills", 
                    "Entertainment", "Uncategorized"]

    def get_merchant_stats(self) -> Dict[str, int]:
        """Get merchant and alias counts."""
        try:
            return {
                "merchant_count": len(self.merchant_categorizer.get_all_merchants()),
                "alias_count": len(self.merchant_categorizer.get_all_aliases())
            }
        except Exception as e:
            logger.error(f"Error getting merchant stats: {str(e)}", exc_info=True)
            return {"merchant_count": 0, "alias_count": 0}

    def has_uncharacterized_merchants(self) -> bool:
        """Check if there are any uncharacterized merchants."""
        try:
            if not os.path.exists(self.review_path):
                return False
            with open(self.review_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    return False
                merchants_data = json.loads(content)
                return len(merchants_data) > 0
        except Exception as e:
            logger.error(f"Error checking uncharacterized merchants: {str(e)}", exc_info=True)
            return False

    def get_uncharacterized_count(self) -> int:
        """Get the count of uncharacterized merchants."""
        try:
            if not os.path.exists(self.review_path):
                return 0
            with open(self.review_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    return 0
                merchants_data = json.loads(content)
                return len(merchants_data)
        except Exception as e:
            logger.error(f"Error getting uncharacterized count: {str(e)}", exc_info=True)
            return 0

    def get_uncharacterized_merchants(self, page: int = 1, limit: int = 10) -> Dict[str, Any]:
        """Get paginated uncharacterized merchants."""
        try:
            if not os.path.exists(self.review_path):
                return {
                    "merchants": [],
                    "total": 0,
                    "page": page,
                    "limit": limit,
                    "pages": 0
                }

            with open(self.review_path, 'r') as f:
                content = f.read().strip()
                merchants_data = json.loads(content) if content else {}

            # Sort by frequency
            sorted_merchants = sorted(merchants_data.items(),
                                   key=lambda x: x[1]["count"],
                                   reverse=True)

            total_merchants = len(sorted_merchants)
            total_pages = (total_merchants + limit - 1) // limit if total_merchants > 0 else 0

            # Adjust page if out of bounds
            if page > total_pages and total_pages > 0:
                page = 1

            # Paginate results
            start = (page - 1) * limit
            end = start + limit
            paginated_merchants = sorted_merchants[start:end] if total_merchants > 0 else []

            # Format for response
            result = []
            for merchant, data in paginated_merchants:
                result.append({
                    "merchant": merchant,
                    "count": data["count"],
                    "total_amount": data["total_amount"],
                    "examples": data["examples"]
                })

            return {
                "merchants": result,
                "total": total_merchants,
                "page": page,
                "limit": limit,
                "pages": total_pages
            }
        except Exception as e:
            logger.error(f"Error getting uncharacterized merchants: {str(e)}", exc_info=True)
            return {
                "merchants": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "pages": 0
            }

    def categorize_uncharacterized_merchant(self, merchant_name: str, category: str) -> bool:
        """Categorize an uncharacterized merchant."""
        try:
            # First verify category exists in databank
            if category not in self.databank.get('categories', {}):
                return False

            # Add merchant to database
            success = self.merchant_categorizer.add_merchant(merchant_name, category)

            if success:
                # Add pattern to databank
                merchant_terms = merchant_name.lower().split()
                pattern = {
                    "terms": merchant_terms,
                    "dateAdded": datetime.now().isoformat(),
                    "lastUpdated": datetime.now().isoformat(),
                    "matchCount": 0
                }
                self.databank['categories'][category]['patterns'].append(pattern)
                self.save_databank()

                # Remove from uncharacterized list
                if os.path.exists(self.review_path):
                    with open(self.review_path, 'r') as f:
                        merchants_data = json.load(f)

                    if merchant_name in merchants_data:
                        del merchants_data[merchant_name]

                    with open(self.review_path, 'w') as f:
                        json.dump(merchants_data, f, indent=2)

            return success
        except Exception as e:
            logger.error(f"Error categorizing merchant: {str(e)}", exc_info=True)
            return False 