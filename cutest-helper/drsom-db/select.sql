WITH ranked_messages AS (SELECT m.*, ROW_NUMBER() OVER (PARTITION BY name,param,method ORDER BY `update` DESC) AS rn
                         FROM cutest.result AS m)
select id,
       n,
       t,
       k,
       method,
       kf,
       kg,
       kh,
       df,
       status
from ranked_messages
where k <= 5000
  and status = 1
  and t <= 100
#   and n <= 200
  and rn = 1
  and `precision` = 1e-5
  and method in ('\\hsodm', '\\lbfgs', '\\newtontr', '\\arc')
and `update`='2023-05-02 17:43:13'
order by n desc, name, t
;