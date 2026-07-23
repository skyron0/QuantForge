from backend.risk.models import RiskContext
from backend.positioning.models import PositionSizingContext

with open("schemas_output.txt", "w") as f:
    f.write("--- RISK CONTEXT ---\n")
    for name, field in RiskContext.__dataclass_fields__.items():
        f.write(f"Name: {name}, Type: {field.type}\n")
    f.write("--- POSITION SIZING CONTEXT ---\n")
    for name, field in PositionSizingContext.__dataclass_fields__.items():
        f.write(f"Name: {name}, Type: {field.type}\n")
