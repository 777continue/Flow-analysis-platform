.PHONY: server nginx locust 
server:
	@sudo node locust/server.js
nginx:
	@sudo nginx -p `pwd`/ -c locust/nginx.conf -s reload
locust:
	@locust -f locust/locustfile.py --host=http://example.com
