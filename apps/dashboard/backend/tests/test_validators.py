# 楠岃瘉鍣ㄦ祴璇?import pytest
from uav_dashboard_api.utils.validators import Validators, sanitize_input, validate_required_fields

class TestValidators:
    """楠岃瘉鍣ㄦ祴璇曠被"""

    def test_validate_username_valid(self):
        """娴嬭瘯鏈夋晥鐨勭敤鎴峰悕"""
        is_valid, error = Validators.validate_username('testuser')
        assert is_valid is True
        assert error is None

    def test_validate_username_too_short(self):
        """娴嬭瘯杩囩煭鐨勭敤鎴峰悕"""
        is_valid, error = Validators.validate_username('ab')
        assert is_valid is False
        assert '闀垮害涓嶈兘灏戜簬3涓瓧绗? in error

    def test_validate_username_too_long(self):
        """娴嬭瘯杩囬暱鐨勭敤鎴峰悕"""
        is_valid, error = Validators.validate_username('a' * 21)
        assert is_valid is False
        assert '闀垮害涓嶈兘瓒呰繃20涓瓧绗? in error

    def test_validate_username_invalid_chars(self):
        """娴嬭瘯鍖呭惈闈炴硶瀛楃鐨勭敤鎴峰悕"""
        is_valid, error = Validators.validate_username('test@user')
        assert is_valid is False
        assert '鍙兘鍖呭惈瀛楁瘝銆佹暟瀛楀拰涓嬪垝绾? in error

    def test_validate_password_valid(self):
        """娴嬭瘯鏈夋晥鐨勫瘑鐮?""
        is_valid, error = Validators.validate_password('password123')
        assert is_valid is True
        assert error is None

    def test_validate_password_too_short(self):
        """娴嬭瘯杩囩煭鐨勫瘑鐮?""
        is_valid, error = Validators.validate_password('12345')
        assert is_valid is False
        assert '闀垮害涓嶈兘灏戜簬6涓瓧绗? in error

    def test_validate_password_too_long(self):
        """娴嬭瘯杩囬暱鐨勫瘑鐮?""
        is_valid, error = Validators.validate_password('a' * 21)
        assert is_valid is False
        assert '闀垮害涓嶈兘瓒呰繃20涓瓧绗? in error

    def test_validate_email_valid(self):
        """娴嬭瘯鏈夋晥鐨勯偖绠?""
        is_valid, error = Validators.validate_email('test@example.com')
        assert is_valid is True
        assert error is None

    def test_validate_email_invalid(self):
        """娴嬭瘯鏃犳晥鐨勯偖绠?""
        is_valid, error = Validators.validate_email('invalid-email')
        assert is_valid is False
        assert '鏍煎紡涓嶆纭? in error

    def test_validate_email_empty(self):
        """娴嬭瘯绌洪偖绠憋紙鍙€夊瓧娈碉級"""
        is_valid, error = Validators.validate_email('')
        assert is_valid is True
        assert error is None

    def test_validate_phone_valid(self):
        """娴嬭瘯鏈夋晥鐨勬墜鏈哄彿"""
        is_valid, error = Validators.validate_phone('13800138000')
        assert is_valid is True
        assert error is None

    def test_validate_phone_invalid(self):
        """娴嬭瘯鏃犳晥鐨勬墜鏈哄彿"""
        is_valid, error = Validators.validate_phone('12345678901')
        assert is_valid is False
        assert '鏍煎紡涓嶆纭? in error

    def test_validate_phone_empty(self):
        """娴嬭瘯绌烘墜鏈哄彿锛堝彲閫夊瓧娈碉級"""
        is_valid, error = Validators.validate_phone('')
        assert is_valid is True
        assert error is None

    def test_validate_battery_level_valid(self):
        """娴嬭瘯鏈夋晥鐨勭數姹犵數閲?""
        is_valid, error = Validators.validate_battery_level(85)
        assert is_valid is True
        assert error is None

    def test_validate_battery_level_too_low(self):
        """娴嬭瘯杩囦綆鐨勭數姹犵數閲?""
        is_valid, error = Validators.validate_battery_level(-10)
        assert is_valid is False
        assert '蹇呴』鍦?-100涔嬮棿' in error

    def test_validate_battery_level_too_high(self):
        """娴嬭瘯杩囬珮鐨勭數姹犵數閲?""
        is_valid, error = Validators.validate_battery_level(150)
        assert is_valid is False
        assert '蹇呴』鍦?-100涔嬮棿' in error

    def test_validate_fault_level_valid(self):
        """娴嬭瘯鏈夋晥鐨勬晠闅滅骇鍒?""
        for level in ['涓ラ噸', '璀﹀憡', '淇℃伅']:
            is_valid, error = Validators.validate_fault_level(level)
            assert is_valid is True
            assert error is None

    def test_validate_fault_level_invalid(self):
        """娴嬭瘯鏃犳晥鐨勬晠闅滅骇鍒?""
        is_valid, error = Validators.validate_fault_level('鏈煡')
        assert is_valid is False
        assert '蹇呴』鏄互涓嬩箣涓€' in error

    def test_validate_fault_status_valid(self):
        """娴嬭瘯鏈夋晥鐨勬晠闅滅姸鎬?""
        for status in ['寰呭鐞?, '澶勭悊涓?, '宸茶В鍐?]:
            is_valid, error = Validators.validate_fault_status(status)
            assert is_valid is True
            assert error is None

    def test_validate_fault_status_invalid(self):
        """娴嬭瘯鏃犳晥鐨勬晠闅滅姸鎬?""
        is_valid, error = Validators.validate_fault_status('鏈煡鐘舵€?)
        assert is_valid is False
        assert '蹇呴』鏄互涓嬩箣涓€' in error


class TestSanitizeInput:
    """杈撳叆娓呯悊娴嬭瘯绫?""

    def test_sanitize_html_string(self):
        """娴嬭瘯娓呯悊HTML瀛楃涓?""
        from uav_dashboard_api.utils.validators import sanitize_html
        input_str = '<script>alert("xss")</script>Hello'
        output = sanitize_html(input_str)
        assert '<script>' not in output
        assert '&lt;script&gt;' in output
        assert 'Hello' in output

    def test_sanitize_input_dict(self):
        """娴嬭瘯娓呯悊瀛楀吀杈撳叆"""
        input_data = {
            'username': 'test',
            'description': '<script>alert("xss")</script>Test'
        }
        output = sanitize_input(input_data)
        assert '<script>' not in output['description']
        assert output['username'] == 'test'

    def test_sanitize_input_list(self):
        """娴嬭瘯娓呯悊鍒楄〃杈撳叆"""
        input_data = [
            {'name': 'test1'},
            {'name': '<script>alert("xss")</script>test2'}
        ]
        output = sanitize_input(input_data)
        assert '<script>' not in output[1]['name']


class TestValidateRequiredFields:
    """蹇呭～瀛楁楠岃瘉娴嬭瘯绫?""

    def test_all_fields_present(self):
        """娴嬭瘯鎵€鏈夊瓧娈甸兘瀛樺湪"""
        data = {'username': 'test', 'password': '123'}
        is_valid, error = validate_required_fields(data, ['username', 'password'])
        assert is_valid is True
        assert error is None

    def test_missing_required_field(self):
        """娴嬭瘯缂哄皯蹇呭～瀛楁"""
        data = {'username': 'test'}
        is_valid, error = validate_required_fields(data, ['username', 'password'])
        assert is_valid is False
        assert 'password' in error

    def test_empty_required_field(self):
        """娴嬭瘯蹇呭～瀛楁涓虹┖"""
        data = {'username': '', 'password': '123'}
        is_valid, error = validate_required_fields(data, ['username', 'password'])
        assert is_valid is False
        assert 'username' in error
