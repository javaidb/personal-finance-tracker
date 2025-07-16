from flask import Blueprint, jsonify, request, render_template
from src.web.services.bank_branding_service import BankBrandingService

bank_branding_bp = Blueprint('bank_branding', __name__, url_prefix='/api/bank-branding')
bank_branding_service = BankBrandingService()

# Create a separate blueprint for web routes
bank_branding_web_bp = Blueprint('bank_branding_web', __name__, url_prefix='/bank-branding')


@bank_branding_bp.route('/<bank_name>', methods=['GET'])
def get_bank_branding(bank_name):
    """Get branding information for a specific bank."""
    try:
        branding_info = bank_branding_service.get_bank_display_info(bank_name)
        return jsonify({
            'success': True,
            'data': branding_info
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@bank_branding_bp.route('/<bank_name>/colors', methods=['GET'])
def get_bank_colors(bank_name):
    """Get color scheme for a specific bank."""
    try:
        colors = bank_branding_service.get_bank_colors(bank_name)
        return jsonify({
            'success': True,
            'data': colors
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@bank_branding_bp.route('/<bank_name>/logo', methods=['GET'])
def get_bank_logo(bank_name):
    """Get logo information for a specific bank."""
    try:
        logo_path = bank_branding_service.get_bank_logo_path(bank_name)
        logo_exists = bank_branding_service.validate_logo_exists(bank_name)
        
        return jsonify({
            'success': True,
            'data': {
                'logo_path': logo_path,
                'logo_exists': logo_exists
            }
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@bank_branding_bp.route('/<bank_name>/theme', methods=['GET'])
def get_bank_theme(bank_name):
    """Get theme information for a specific bank."""
    try:
        theme = bank_branding_service.get_bank_theme(bank_name)
        theme_class = bank_branding_service.get_theme_class_name(bank_name)
        gradients = bank_branding_service.get_bank_gradients(bank_name)
        
        return jsonify({
            'success': True,
            'data': {
                'theme': theme,
                'theme_class': theme_class,
                'gradients': gradients
            }
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@bank_branding_bp.route('/all', methods=['GET'])
def get_all_bank_branding():
    """Get branding information for all available banks."""
    try:
        all_branding = {}
        for bank_name in bank_branding_service.bank_config.get_available_banks():
            all_branding[bank_name] = bank_branding_service.get_bank_display_info(bank_name)
        
        return jsonify({
            'success': True,
            'data': all_branding
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@bank_branding_bp.route('/available', methods=['GET'])
def get_available_banks():
    """Get list of available banks with basic branding info."""
    try:
        available_banks = bank_branding_service.bank_config.get_available_banks()
        banks_info = []
        
        for bank_name in available_banks:
            display_name = bank_branding_service.bank_config.get_bank_display_name(bank_name)
            logo_path = bank_branding_service.get_bank_logo_path(bank_name)
            theme_class = bank_branding_service.get_theme_class_name(bank_name)
            
            banks_info.append({
                'name': bank_name,
                'display_name': display_name,
                'logo_path': logo_path,
                'theme_class': theme_class
            })
        
        return jsonify({
            'success': True,
            'data': banks_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500 


@bank_branding_web_bp.route('/demo')
def bank_branding_demo():
    """Demo page for bank branding system."""
    return render_template('bank_branding_demo.html') 