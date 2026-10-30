import urllib.request
c = urllib.request.urlopen("http://localhost:9529/src/views/home/index.vue").read().decode('utf-8')
print("LEN", len(c))
print("HAS_VELSEIF", "v-else-if" in c)
print("STUDENT_VIF", c.count('role === "student"'))
print("DEBUG_LOG", "HOME-DEBUG" in c)
print("ROLE_SPAN", "role={{ role }}" in c)
i = c.find('role === "student"')
print("CTX", c[i-120:i+60] if i>=0 else "NONE")
