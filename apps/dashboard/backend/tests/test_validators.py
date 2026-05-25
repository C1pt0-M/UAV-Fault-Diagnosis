from uav_dashboard_api.utils.validators import (
    Validators,
    sanitize_input,
    validate_required_fields,
)


class TestValidators:
    def test_validate_username_valid(self):
        is_valid, error = Validators.validate_username("testuser")
        assert is_valid is True
        assert error is None

    def test_validate_username_too_short(self):
        is_valid, error = Validators.validate_username("ab")
        assert is_valid is False
        assert "长度不能少于3个字符" in error

    def test_validate_username_too_long(self):
        is_valid, error = Validators.validate_username("a" * 21)
        assert is_valid is False
        assert "长度不能超过20个字符" in error

    def test_validate_username_invalid_chars(self):
        is_valid, error = Validators.validate_username("test@user")
        assert is_valid is False
        assert "只能包含字母、数字和下划线" in error

    def test_validate_password_valid(self):
        is_valid, error = Validators.validate_password("password123")
        assert is_valid is True
        assert error is None

    def test_validate_password_too_short(self):
        is_valid, error = Validators.validate_password("12345")
        assert is_valid is False
        assert "长度不能少于6个字符" in error

    def test_validate_password_too_long(self):
        is_valid, error = Validators.validate_password("a" * 21)
        assert is_valid is False
        assert "长度不能超过20个字符" in error

    def test_validate_email_valid(self):
        is_valid, error = Validators.validate_email("test@example.com")
        assert is_valid is True
        assert error is None

    def test_validate_email_invalid(self):
        is_valid, error = Validators.validate_email("invalid-email")
        assert is_valid is False
        assert "格式不正确" in error

    def test_validate_email_empty(self):
        is_valid, error = Validators.validate_email("")
        assert is_valid is True
        assert error is None

    def test_validate_phone_valid(self):
        is_valid, error = Validators.validate_phone("13800138000")
        assert is_valid is True
        assert error is None

    def test_validate_phone_invalid(self):
        is_valid, error = Validators.validate_phone("12345678901")
        assert is_valid is False
        assert "格式不正确" in error

    def test_validate_phone_empty(self):
        is_valid, error = Validators.validate_phone("")
        assert is_valid is True
        assert error is None

    def test_validate_battery_level_valid(self):
        is_valid, error = Validators.validate_battery_level(85)
        assert is_valid is True
        assert error is None

    def test_validate_battery_level_too_low(self):
        is_valid, error = Validators.validate_battery_level(-10)
        assert is_valid is False
        assert "必须在0-100之间" in error

    def test_validate_battery_level_too_high(self):
        is_valid, error = Validators.validate_battery_level(150)
        assert is_valid is False
        assert "必须在0-100之间" in error

    def test_validate_fault_level_valid(self):
        for level in ["严重", "警告", "信息"]:
            is_valid, error = Validators.validate_fault_level(level)
            assert is_valid is True
            assert error is None

    def test_validate_fault_level_invalid(self):
        is_valid, error = Validators.validate_fault_level("未知")
        assert is_valid is False
        assert "必须是以下之一" in error

    def test_validate_fault_status_valid(self):
        for status in ["待处理", "处理中", "已解决"]:
            is_valid, error = Validators.validate_fault_status(status)
            assert is_valid is True
            assert error is None

    def test_validate_fault_status_invalid(self):
        is_valid, error = Validators.validate_fault_status("未知状态")
        assert is_valid is False
        assert "必须是以下之一" in error


class TestSanitizeInput:
    def test_sanitize_html_string(self):
        from uav_dashboard_api.utils.validators import sanitize_html

        input_str = '<script>alert("xss")</script>Hello'
        output = sanitize_html(input_str)
        assert "<script>" not in output
        assert "&lt;script&gt;" in output
        assert "Hello" in output

    def test_sanitize_input_dict(self):
        input_data = {
            "username": "test",
            "description": '<script>alert("xss")</script>Test',
        }
        output = sanitize_input(input_data)
        assert "<script>" not in output["description"]
        assert output["username"] == "test"

    def test_sanitize_input_list(self):
        input_data = [
            {"name": "test1"},
            {"name": '<script>alert("xss")</script>test2'},
        ]
        output = sanitize_input(input_data)
        assert "<script>" not in output[1]["name"]


class TestValidateRequiredFields:
    def test_all_fields_present(self):
        data = {"username": "test", "password": "123"}
        is_valid, error = validate_required_fields(data, ["username", "password"])
        assert is_valid is True
        assert error is None

    def test_missing_required_field(self):
        data = {"username": "test"}
        is_valid, error = validate_required_fields(data, ["username", "password"])
        assert is_valid is False
        assert "password" in error

    def test_empty_required_field(self):
        data = {"username": "", "password": "123"}
        is_valid, error = validate_required_fields(data, ["username", "password"])
        assert is_valid is False
        assert "username" in error
