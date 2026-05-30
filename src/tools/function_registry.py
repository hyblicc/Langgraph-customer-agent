"""
Function Calling registry and management
"""

import logging
import json
from typing import Dict, Any, Callable, Optional, List
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


class FunctionRegistry:
    """
    Registry for dynamically callable tools/functions
    
    Implements the Function Calling pattern for LLM integration
    """
    
    def __init__(self):
        """Initialize function registry"""
        self._functions: Dict[str, Dict[str, Any]] = {}
        self._call_history: List[Dict[str, Any]] = []
        logger.info("FunctionRegistry initialized")
    
    def register(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Register a function for dynamic calling
        
        Args:
            name: Function identifier
            func: Callable function
            description: Human-readable description
            parameters: JSON Schema for parameters
        
        Returns:
            Registered function (unchanged)
        """
        logger.info(f"Registering function: {name}")
        
        # Default parameter schema
        if parameters is None:
            parameters = {
                "type": "object",
                "properties": {},
                "required": [],
            }
        
        self._functions[name] = {
            "func": func,
            "description": description,
            "parameters": parameters,
            "registered_at": datetime.now().isoformat(),
        }
        
        logger.debug(f"Function registered: {name}")
        return func
    
    def call(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a registered function
        
        Args:
            name: Function name
            **kwargs: Function arguments
        
        Returns:
            Execution result
        """
        logger.info(f"Calling function: {name} with args: {list(kwargs.keys())}")
        
        if name not in self._functions:
            error_msg = f"Function not found: {name}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "result": None,
            }
        
        try:
            func_info = self._functions[name]
            func = func_info["func"]
            
            # Execute function
            result = func(**kwargs)
            
            # Log call
            self._call_history.append({
                "function": name,
                "arguments": kwargs,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "success": True,
            })
            
            logger.info(f"Function {name} executed successfully")
            
            return {
                "success": True,
                "result": result,
                "error": None,
            }
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error calling function {name}: {error_msg}")
            
            # Log failed call
            self._call_history.append({
                "function": name,
                "arguments": kwargs,
                "result": None,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": error_msg,
            })
            
            return {
                "success": False,
                "error": error_msg,
                "result": None,
            }
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions in OpenAI format
        
        Returns:
            List of tool definitions compatible with OpenAI API
        """
        tools = []
        
        for name, info in self._functions.items():
            tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info.get("description", ""),
                    "parameters": info.get("parameters", {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }),
                }
            }
            tools.append(tool)
        
        logger.debug(f"Generated {len(tools)} OpenAI tool definitions")
        return tools
    
    def list_functions(self) -> List[str]:
        """List all registered functions"""
        return list(self._functions.keys())
    
    def get_function_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered function"""
        return self._functions.get(name)
    
    def get_call_history(self, function_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get call history, optionally filtered by function name"""
        if function_name:
            return [c for c in self._call_history if c["function"] == function_name]
        return self._call_history
    
    def clear_history(self) -> None:
        """Clear call history"""
        self._call_history = []
        logger.info("Call history cleared")


# Global registry instance
_registry = FunctionRegistry()


def register_tool(
    name: str,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator for registering a function as a tool
    
    Usage:
        @register_tool(name="get_weather", description="Get weather info")
        def get_weather(city: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        _registry.register(name, func, description, parameters)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def get_registry() -> FunctionRegistry:
    """Get the global function registry"""
    return _registry


# ============================================================================
# Example Tool Registrations
# ============================================================================

@register_tool(
    name="get_order_status",
    description="Get the status of a customer order",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID to check"
            }
        },
        "required": ["order_id"]
    }
)
def get_order_status(order_id: str) -> Dict[str, Any]:
    """Get order status (mock implementation)"""
    logger.info(f"Getting order status for: {order_id}")
    
    # Mock data
    return {
        "order_id": order_id,
        "status": "shipped",
        "tracking_number": "TRK-123456",
        "estimated_delivery": "2026-06-05",
        "items": [
            {"name": "Product A", "quantity": 2},
            {"name": "Product B", "quantity": 1},
        ]
    }


@register_tool(
    name="check_inventory",
    description="Check product inventory",
    parameters={
        "type": "object",
        "properties": {
            "product_id": {
                "type": "string",
                "description": "The product ID to check"
            }
        },
        "required": ["product_id"]
    }
)
def check_inventory(product_id: str) -> Dict[str, Any]:
    """Check product inventory (mock implementation)"""
    logger.info(f"Checking inventory for product: {product_id}")
    
    # Mock data
    return {
        "product_id": product_id,
        "stock_available": 150,
        "in_stock": True,
        "reorder_level": 50,
    }


@register_tool(
    name="process_return",
    description="Process a return request",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID to return"
            },
            "reason": {
                "type": "string",
                "description": "Reason for return"
            }
        },
        "required": ["order_id", "reason"]
    }
)
def process_return(order_id: str, reason: str) -> Dict[str, Any]:
    """Process return request (mock implementation)"""
    logger.info(f"Processing return for order {order_id}: {reason}")
    
    # Mock data
    return {
        "return_id": f"RET-{order_id}",
        "status": "approved",
        "refund_amount": 99.99,
        "shipping_label": "https://example.com/label.pdf",
    }


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Get registry
    registry = get_registry()
    
    # List registered functions
    print("Registered functions:")
    for func_name in registry.list_functions():
        print(f"  - {func_name}")
    
    # Test function call
    print("\nTesting function call:")
    result = registry.call("get_order_status", order_id="ORD-12345")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Get OpenAI tools
    print("\nOpenAI Tools Format:")
    tools = registry.get_openai_tools()
    print(json.dumps(tools, indent=2))
