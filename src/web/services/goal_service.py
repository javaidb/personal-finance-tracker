import json
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
from ..models.goal import Goal
import logging

logger = logging.getLogger(__name__)

class GoalService:
    """Service class for handling goal-related operations."""

    # Predefined goals (with default target dates 1 year from now)
    @staticmethod
    def get_predefined_goals():
        """Get predefined goals with default target dates."""
        one_year_from_now = datetime.now() + timedelta(days=365)
        return [
            {
                "id": "predefined_motorcycle",
                "name": "Motorcycle",
                "amount": 5000,
                "icon": "motorcycle",
                "target_date": one_year_from_now.isoformat(),
                "is_predefined": True
            },
            {
                "id": "predefined_vietnam",
                "name": "Vietnam Trip",
                "amount": 10000,
                "icon": "plane",
                "target_date": one_year_from_now.isoformat(),
                "is_predefined": True
            }
        ]

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.goals_file = base_path / 'cached_data' / 'goals.json'
        self.ensure_goals_file_exists()

    def ensure_goals_file_exists(self):
        """Ensure the goals.json file exists."""
        try:
            # Create cached_data directory if it doesn't exist
            self.goals_file.parent.mkdir(parents=True, exist_ok=True)

            if not self.goals_file.exists():
                # Initialize with empty custom goals and hidden predefined goals
                initial_data = {
                    "custom_goals": [],
                    "hidden_predefined_goals": []
                }
                with open(self.goals_file, 'w') as f:
                    json.dump(initial_data, f, indent=2)
                logger.info(f"Created goals file at {self.goals_file}")
        except Exception as e:
            logger.error(f"Error creating goals file: {str(e)}", exc_info=True)

    def load_goals(self) -> dict:
        """Load goals from JSON file."""
        try:
            with open(self.goals_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading goals: {str(e)}", exc_info=True)
            return {"custom_goals": []}

    def save_goals(self, goals_data: dict):
        """Save goals to JSON file."""
        try:
            with open(self.goals_file, 'w') as f:
                json.dump(goals_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving goals: {str(e)}", exc_info=True)

    def get_all_goals(self) -> List[dict]:
        """Get all goals (predefined + custom goals combined, excluding hidden)."""
        try:
            data = self.load_goals()
            hidden_predefined = data.get('hidden_predefined_goals', [])

            # Start with predefined goals, excluding hidden ones
            all_goals = [g for g in self.get_predefined_goals() if g['id'] not in hidden_predefined]

            # Add custom goals on top
            custom_goals = data.get('custom_goals', [])
            all_goals.extend(custom_goals)

            return all_goals
        except Exception as e:
            logger.error(f"Error getting all goals: {str(e)}", exc_info=True)
            return self.get_predefined_goals()

    def get_milestone_amounts(self) -> List[float]:
        """Get milestone amounts as a list of floats sorted in ascending order."""
        goals = self.get_all_goals()
        amounts = sorted([goal['amount'] for goal in goals])
        return amounts

    def add_goal(self, name: str, amount: float, target_date: datetime = None, icon: str = None) -> Optional[dict]:
        """Add a new custom goal."""
        try:
            goal = Goal(name=name, amount=amount, target_date=target_date, icon=icon)
            data = self.load_goals()

            # Initialize custom_goals if it doesn't exist
            if 'custom_goals' not in data:
                data['custom_goals'] = []

            # Add the new goal
            data['custom_goals'].append(goal.to_dict())
            self.save_goals(data)

            return goal.to_dict()
        except Exception as e:
            logger.error(f"Error adding goal: {str(e)}", exc_info=True)
            return None

    def update_goal(self, goal_id: str, name: str = None, amount: float = None, target_date: datetime = None, icon: str = None) -> Optional[dict]:
        """Update an existing goal (custom or predefined)."""
        try:
            data = self.load_goals()
            custom_goals = data.get('custom_goals', [])

            # Check if this is a predefined goal being edited
            if goal_id.startswith('predefined_'):
                # Check if we already have a custom version of this predefined goal
                existing_custom = next((g for g in custom_goals if g.get('id') == goal_id), None)

                if existing_custom:
                    # Update the existing custom version
                    for i, goal in enumerate(custom_goals):
                        if goal.get('id') == goal_id:
                            if name is not None:
                                custom_goals[i]['name'] = name
                            if amount is not None:
                                custom_goals[i]['amount'] = amount
                            if target_date is not None:
                                custom_goals[i]['target_date'] = target_date.isoformat() if isinstance(target_date, datetime) else target_date
                            if icon is not None:
                                custom_goals[i]['icon'] = icon
                            data['custom_goals'] = custom_goals
                            self.save_goals(data)
                            return custom_goals[i]
                else:
                    # First time editing this predefined goal - convert to custom
                    predefined_goals = self.get_predefined_goals()
                    predefined_goal = next((g for g in predefined_goals if g['id'] == goal_id), None)

                    if predefined_goal:
                        # Hide the original predefined goal
                        hidden_predefined = data.get('hidden_predefined_goals', [])
                        if goal_id not in hidden_predefined:
                            hidden_predefined.append(goal_id)
                            data['hidden_predefined_goals'] = hidden_predefined

                        # Create a custom goal with the SAME ID to replace it
                        new_goal_dict = {
                            'id': goal_id,  # Keep the same ID!
                            'name': name if name is not None else predefined_goal['name'],
                            'amount': amount if amount is not None else predefined_goal['amount'],
                            'icon': icon if icon is not None else predefined_goal.get('icon', 'piggy-bank'),
                            'target_date': (target_date.isoformat() if isinstance(target_date, datetime) else target_date) if target_date is not None else predefined_goal['target_date'],
                            'created_at': datetime.now().isoformat()
                        }
                        custom_goals.append(new_goal_dict)
                        data['custom_goals'] = custom_goals
                        self.save_goals(data)
                        return new_goal_dict

            # Find the custom goal to update
            goal_found = False
            for i, goal in enumerate(custom_goals):
                if goal.get('id') == goal_id:
                    # Update fields if provided
                    if name is not None:
                        custom_goals[i]['name'] = name
                    if amount is not None:
                        custom_goals[i]['amount'] = amount
                    if target_date is not None:
                        custom_goals[i]['target_date'] = target_date.isoformat() if isinstance(target_date, datetime) else target_date
                    if icon is not None:
                        custom_goals[i]['icon'] = icon

                    goal_found = True
                    data['custom_goals'] = custom_goals
                    self.save_goals(data)
                    return custom_goals[i]

            return None if not goal_found else {}
        except Exception as e:
            logger.error(f"Error updating goal: {str(e)}", exc_info=True)
            return None

    def delete_goal(self, goal_id: str) -> bool:
        """Delete a goal by ID (custom or predefined)."""
        try:
            data = self.load_goals()
            custom_goals = data.get('custom_goals', [])

            # If it's a predefined goal
            if goal_id.startswith('predefined_'):
                # First, check if there's a custom version to delete
                has_custom_version = any(g.get('id') == goal_id for g in custom_goals)

                if has_custom_version:
                    # Remove the custom version
                    updated_goals = [g for g in custom_goals if g.get('id') != goal_id]
                    data['custom_goals'] = updated_goals
                    # Also remove from hidden list to restore the original predefined
                    hidden_predefined = data.get('hidden_predefined_goals', [])
                    if goal_id in hidden_predefined:
                        hidden_predefined.remove(goal_id)
                        data['hidden_predefined_goals'] = hidden_predefined
                    self.save_goals(data)
                    return True
                else:
                    # No custom version, just hide the predefined goal
                    hidden_predefined = data.get('hidden_predefined_goals', [])
                    if goal_id not in hidden_predefined:
                        hidden_predefined.append(goal_id)
                        data['hidden_predefined_goals'] = hidden_predefined
                        self.save_goals(data)
                        return True
                    return False

            # Otherwise, delete from custom goals
            updated_goals = [g for g in custom_goals if g.get('id') != goal_id]

            if len(updated_goals) == len(custom_goals):
                # No goal was deleted
                return False

            data['custom_goals'] = updated_goals
            self.save_goals(data)
            return True
        except Exception as e:
            logger.error(f"Error deleting goal: {str(e)}", exc_info=True)
            return False

    def has_custom_goals(self) -> bool:
        """Check if there are any custom goals."""
        try:
            data = self.load_goals()
            return len(data.get('custom_goals', [])) > 0
        except Exception as e:
            logger.error(f"Error checking custom goals: {str(e)}", exc_info=True)
            return False
