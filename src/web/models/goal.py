from datetime import datetime

class Goal:
    """Goal model for representing financial goals/milestones."""

    def __init__(self, name: str, amount: float, target_date: datetime = None, icon: str = None, goal_id: str = None, created_at: datetime = None):
        self.name = name
        self.amount = amount
        self.target_date = target_date
        self.icon = icon or 'piggy-bank'  # Default icon
        self.goal_id = goal_id or f"{name.lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"
        self.created_at = created_at or datetime.now()

    def to_dict(self):
        """Convert goal to dictionary."""
        return {
            'id': self.goal_id,
            'name': self.name,
            'amount': self.amount,
            'icon': self.icon,
            'target_date': self.target_date.isoformat() if isinstance(self.target_date, datetime) else self.target_date,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create a Goal instance from a dictionary."""
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        target_date = data.get('target_date')
        if isinstance(target_date, str):
            target_date = datetime.fromisoformat(target_date)

        return cls(
            name=data['name'],
            amount=data['amount'],
            target_date=target_date,
            icon=data.get('icon'),
            goal_id=data.get('id'),
            created_at=created_at
        )
