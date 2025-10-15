# file: tools.py (Upgraded Version)

from langchain.tools import tool
from pydantic.v1 import BaseModel, Field
from typing import Optional

class WorkOrderInput(BaseModel):
    """Input schema for the create_work_order tool."""
    location: str = Field(description="The primary location of the issue, like a street name, highway number, or address.")
    issue: str = Field(description="A detailed description of the maintenance issue being reported.")
    priority: int = Field(description="The priority of the request, where 1 is high, 2 is medium, and 3 is low.", ge=1, le=3)
    direction: Optional[str] = Field(
        default=None, 
        description="The direction of travel if on a highway (e.g., westbound, northbound, southbound, eastbound)."
    )
    landmark: Optional[str] = Field(
        default=None, 
        description="A nearby landmark to help crews find the exact location (e.g., 'near the big oak tree', 'just past the McDonalds')."
    )

@tool("create_work_order", args_schema=WorkOrderInput)
def create_work_order(location: str, issue: str, priority: int, direction: Optional[str] = None, landmark: Optional[str] = None) -> str:
    """
    Use this tool when a user reports a specific maintenance issue that needs to be fixed.
    This tool requires a detailed location, a description of the issue, and a priority.
    It can also accept optional direction and landmark details for more precision.
    """
    # In a real application, you would save these details to a database.
    # The document would now look much richer.
    report_details = {
        "location": location,
        "issue": issue,
        "priority": priority,
        "direction": direction,
        "landmark": landmark,
        "status": "open"
    }
    
    print(f"--- Creating detailed work order: {report_details} ---")
    work_order_id = "WO-2025-1014-C8D9" # Generate a unique ID
    
    # Create a confirmation message
    confirmation = f"Successfully created work order {work_order_id} for a priority-{priority} issue: '{issue}' at {location}."
    if direction:
        confirmation += f" (Direction: {direction})"
    if landmark:
        confirmation += f" (Landmark: {landmark})"
    confirmation += " A crew will be dispatched."
    
    return confirmation