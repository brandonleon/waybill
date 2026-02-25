install:
  uv tool install --no-cache .

uninstall:
  uv tool uninstall waybill

upgrade:
  just uninstall
  just install
