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
        self.review_path = os.path.join(base_path, 'cached_data', 'uncategorized_merchants.json')
        self.category_colors_path = os.path.join(base_path, 'cached_data', 'category_colors.json')
        self.load_databank()

    def load_databank(self):
        """Load or create databank.json"""
        try:
            if os.path.exists(self.databank_path):
                with open(self.databank_path, 'r') as f:
                    self.databank = json.load(f)
            else:
                # On first run, load categories from category_colors.json
                if os.path.exists(self.category_colors_path):
                    with open(self.category_colors_path, 'r') as f:
                        category_colors = json.load(f)
                        # Create initial databank structure using categories from category_colors.json
                        self.databank = {
                            "categories": {
                                category: {
                                    "totalMatches": 0,
                                    "patterns": []
                                } for category in category_colors.keys()
                            }
                        }
                else:
                    # If category_colors.json doesn't exist, use minimal default categories
                    self.databank = {
                        "categories": {
                            "Uncategorized": {"totalMatches": 0, "patterns": []}
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
            logger.debug(f"Raw merchants from categorizer: {merchants}")
            
            if not merchants:
                logger.warning("No merchants found in database")
                return []
            
            formatted_merchants = []
            for name, category in merchants.items():
                if name != "metadata":  # Skip metadata entries
                    formatted_merchants.append({
                        "name": name,
                        "category": category
                    })
            
            logger.debug(f"Formatted merchants: {formatted_merchants}")
            return formatted_merchants
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
            logger.debug(f"Adding merchant: {merchant_name} with category: {category}")
            logger.debug(f"Available categories: {list(self.databank.get('categories', {}).keys())}")
            
            if category not in self.databank.get('categories', {}):
                logger.error(f"Category {category} not found in databank")
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
                logger.debug(f"Successfully added merchant {merchant_name} to category {category}")
                
                # Trigger transaction recategorization using global service
                from ..routes.api import init_transaction_service
                transaction_service = init_transaction_service()
                if transaction_service:
                    transaction_service.recategorize_transactions()
                else:
                    logger.warning("Could not initialize transaction service for recategorization")
            else:
                logger.error(f"Failed to add merchant {merchant_name} to database")
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
            merchants = self.merchant_categorizer.get_all_merchants()
            aliases = self.merchant_categorizer.get_all_aliases()
            
            if merchants is None:
                logger.error("Failed to get merchants from categorizer")
                return {"merchant_count": 0, "alias_count": 0}
            
            if aliases is None:
                logger.error("Failed to get aliases from categorizer")
                return {"merchant_count": len(merchants), "alias_count": 0}
            
            return {
                "merchant_count": len(merchants),
                "alias_count": len(aliases)
            }
        except Exception as e:
            logger.error(f"Error getting merchant stats: {str(e)}", exc_info=True)
            return {"merchant_count": 0, "alias_count": 0}

    def has_uncategorized_merchants(self) -> bool:
        """Check if there are any uncategorized merchants."""
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
            logger.error(f"Error checking uncategorized merchants: {str(e)}", exc_info=True)
            return False

    def get_uncategorized_count(self) -> int:
        """Get the count of uncategorized merchants."""
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
            logger.error(f"Error getting uncategorized count: {str(e)}", exc_info=True)
            return 0

    def get_uncategorized_merchants(self, page: int = 1, limit: int = 10) -> Dict[str, Any]:
        """Get paginated uncategorized merchants."""
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
            logger.error(f"Error getting uncategorized merchants: {str(e)}", exc_info=True)
            return {
                "merchants": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "pages": 0
            }

    def categorize_uncategorized_merchant(self, merchant_name: str, category: str) -> bool:
        """Categorize an uncategorized merchant."""
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

                # Remove from uncategorized list
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

    def add_category(self, category_name: str) -> bool:
        """Add a new category."""
        try:
            if category_name in self.databank.get('categories', {}):
                return False
            
            self.databank['categories'][category_name] = {
                "totalMatches": 0,
                "patterns": []
            }
            self.save_databank()
            return True
        except Exception as e:
            logger.error(f"Error adding category: {str(e)}", exc_info=True)
            return False

    def rename_category(self, old_name: str, new_name: str) -> bool:
        """Rename a category."""
        try:
            if old_name not in self.databank.get('categories', {}) or new_name in self.databank.get('categories', {}):
                return False
            
            # Get all merchants with this category
            merchants = self.get_all_merchants()
            affected_merchants = [m['name'] for m in merchants if m['category'] == old_name]
            
            # Update category in databank
            self.databank['categories'][new_name] = self.databank['categories'].pop(old_name)
            self.save_databank()
            
            # Update all merchants with this category
            for merchant_name in affected_merchants:
                self.merchant_categorizer.add_merchant(merchant_name, new_name)
            
            return True
        except Exception as e:
            logger.error(f"Error renaming category: {str(e)}", exc_info=True)
            return False

    def delete_category(self, category_name: str) -> bool:
        """Delete a category."""
        try:
            # Don't allow deletion of the Uncategorized category
            if category_name.lower() == "uncategorized":
                logger.error("Cannot delete the Uncategorized category as it is required by the system")
                return False
            
            # Case-insensitive check for category existence
            category_actual_name = None
            for cat in self.databank.get('categories', {}):
                if cat.lower() == category_name.lower():
                    category_actual_name = cat
                    break
            
            if not category_actual_name:
                logger.error(f"Category {category_name} not found")
                return False
            
            # Get all merchants with this category
            merchants = self.get_all_merchants()
            affected_merchants = [m['name'] for m in merchants if m['category'].lower() == category_name.lower()]
            
            # Delete category from databank
            del self.databank['categories'][category_actual_name]
            self.save_databank()
            
            # Update all merchants with this category to "Uncategorized"
            success = True
            for merchant_name in affected_merchants:
                if not self.merchant_categorizer.add_merchant(merchant_name, "Uncategorized"):
                    logger.error(f"Failed to update merchant {merchant_name} to Uncategorized")
                    success = False
            
            if not success:
                logger.warning("Some merchants could not be updated to Uncategorized")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting category: {str(e)}", exc_info=True)
            return False

    def categorize_transaction(self, processed_details: Dict[str, Any]) -> str:
        """Categorize a transaction based on its details."""
        try:
            # Try merchant-based categorization
            category, _ = self.merchant_categorizer.categorize_transaction(processed_details)
            
            # If merchant categorization found something, use it
            if category != "Uncategorized":
                return category
            
            # If no merchant match and already has a non-Uncategorized category, preserve it
            current_category = processed_details.get('category', 'Uncategorized')
            if current_category != 'Uncategorized':
                return current_category
            
            return 'Uncategorized'
        except Exception as e:
            logger.error(f"Error categorizing transaction: {str(e)}", exc_info=True)
            return 'Uncategorized' 