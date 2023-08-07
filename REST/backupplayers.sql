
--
-- Table structure for table `Players`
--

DROP TABLE IF EXISTS `Players`;

CREATE TABLE `Players` (
  `player_id` int(11) NOT NULL AUTO_INCREMENT,
  `team_id` int(11) DEFAULT NULL,
  `surname` varchar(255) NOT NULL,
  `given_names` varchar(255) NOT NULL,
  `nationality` varchar(255) NOT NULL,
  `date_of_birth` date NOT NULL,
  PRIMARY KEY (`player_id`),
  KEY `team_id` (`team_id`),
  CONSTRAINT `Players_ibfk_1` FOREIGN KEY (`team_id`) REFERENCES `Teams` (`team_id`)
) ENGINE=InnoDB AUTO_INCREMENT=37 DEFAULT CHARSET=latin1;


--
-- Dumping data for table `Players`
--

LOCK TABLES `Players` WRITE;

INSERT INTO `Players` VALUES (1,1,'Smith','John','USA','1990-05-15'),(2,1,'Johnson','Michael','USA','1992-09-20'),(3,1,'Williams','Emily','Canada','1994-07-10'),(4,2,'Brown','David','USA','1988-03-25'),(5,2,'Davis','Jessica','USA','1991-11-03'),(6,2,'Miller','Andrew','Canada','1993-06-18'),(7,3,'Wilson','Sarah','USA','1995-12-08'),(8,3,'Anderson','Daniel','England','1989-02-28'),(9,3,'Taylor','Olivia','USA','1997-04-13');

UNLOCK TABLES;
