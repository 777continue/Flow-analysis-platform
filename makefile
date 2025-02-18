.PHONY: server nginx locust 
server:
	@sudo node locust/server.js
nginx:
	@sudo nginx -p `pwd`/ -c envoy/conf/nginx.conf -s reload
locust:
	@locust -f locust/locustfile.py --host=http://example.com
