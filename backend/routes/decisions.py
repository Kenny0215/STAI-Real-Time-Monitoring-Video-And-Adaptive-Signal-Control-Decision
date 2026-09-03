from flask import Blueprint, jsonify
from fuzzy_controller import get_recent_fuzzy_decisions

decisions_bp = Blueprint("decisions", __name__)


@decisions_bp.route("/api/decisions", methods=["GET"])
def list_decisions():
    """Returns the most recent AI signal control decisions."""
    decisions = get_recent_fuzzy_decisions(limit=50)
    return jsonify({
        "count": len(decisions),
        "data":  decisions,
    })